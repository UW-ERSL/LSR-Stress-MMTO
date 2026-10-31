from networks import VariationalAutoencoder
import torch
from torch.func import vmap, jacrev
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import os
import sys
import warnings
from databases import attribute_name, normalize, participation_ratio

class MaterialEncoder:
    def __init__(self, vae_params):
        self.nAttributes = 0
        self.vae_params = vae_params

    def readExcel(self, excel_file):
        self.excel_file = excel_file
        self.preprocessData()

        self.nMaterials = self.scaledMaterialData.shape[0]
        self.nAttributes = self.scaledMaterialData.shape[1]

        self.vaeSettings = {
            'encoder': {
                'inputDim': self.nAttributes,
                'hiddenDim': self.vae_params.vae_hiddenDim,
                'latentDim': self.vae_params.latentDim
            },
            'decoder': {
                'latentDim': self.vae_params.latentDim,
                'hiddenDim': self.vae_params.vae_hiddenDim,
                'outputDim': self.nAttributes
            }
        }
        self.vaeNet = VariationalAutoencoder(self.vaeSettings)

    def preprocessData(self):
        with warnings.catch_warnings():
            # the databases carry Excel data validation, which openpyxl drops
            warnings.filterwarnings("ignore", message="Data Validation extension")
            df = pd.read_excel(self.excel_file, header=None)
        attribute_names = [attribute_name(n) for n in df.iloc[0, 1:].tolist()]   # aliases -> registry names
        units = df.iloc[1, 1:].tolist()
        material_names = df.iloc[2:, 0].tolist()
        values = df.iloc[2:, 1:].to_numpy(dtype=float)

        self.rawData = values

        normalizedData, dataScaleMin, dataScaleMax = normalize(values, attribute_names)

        scaledMaterialData = torch.tensor(normalizedData, dtype=torch.get_default_dtype())

        materialAttributes = {}
        for i, name in enumerate(attribute_names):
            materialAttributes[name] = {
                'idx': i,
                'unit': units[i],
                'scaleMin': dataScaleMin[i],
                'scaleMax': dataScaleMax[i],
            }

        self.materialNames = material_names
        self.scaledMaterialData = scaledMaterialData
        self.materialAttributes = materialAttributes

        attribute_means = normalizedData.mean(axis=0)
        attribute_stds = normalizedData.std(axis=0)
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        self.computeParticipationRatio()
        print(f"Intrinsic dimension estimate (participation ratio): {self.intrinsic_dim:.3f}")
        print(f"PCA explained variance ratio: {self.pca_explained_variance_ratio}")
    def computeParticipationRatio(self):
        """
        Effective intrinsic dimensionality estimate from PCA eigenvalues.
        Uses normalized data already created in preprocessData.
        """
        X = self.scaledMaterialData.detach().cpu().numpy()
        self.intrinsic_dim, self.pca_explained_variance, self.pca_explained_variance_ratio = \
            participation_ratio(X)
        return self.intrinsic_dim


    def largestEncodingErrorPercent(self):
        self.vaeNet.encoder.isTraining = False
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            diffMatrix = self.scaledMaterialData - decoded
            largestErrorPercent = 100.0 * torch.max(torch.abs(diffMatrix)).item()
        self.vaeNet.encoder.isTraining = True
        return largestErrorPercent

    def printEncodingErrors(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            diffMatrix = self.scaledMaterialData - decoded

        attribute_names = list(self.materialAttributes.keys())
        print("-" * 40)
        print("{:<25} {:>15}".format("Attribute", "Max % Error"))
        print("-" * 40)
        col = 0
        for name in attribute_names:
            percent_err = 100 * diffMatrix[:, col].abs().numpy()
            max_percent_err = np.max(percent_err)
            print("{:<25} {:>15.6f}".format(name, max_percent_err))
            col += 1
        print("-" * 40)
    def compute_parameter_norm(self):
        total_sq = 0.0
        with torch.no_grad():
            for p in self.vaeNet.parameters():
                total_sq += torch.sum(p.detach() ** 2).item()
        return total_sq ** 0.5
    def loadAutoencoderFromFile(self, fileName):
        self.vaeNet.load_state_dict(torch.load(fileName))
        self.vaeNet.eval()
        self.training_latents = self.vaeNet.encoder(self.scaledMaterialData).cpu()
    def _compute_vae_loss(self, klFactor):
        predData = self.vaeNet(self.scaledMaterialData)
        klLoss = klFactor * self.vaeNet.encoder.kl
        reconLoss = ((self.scaledMaterialData - predData) ** 2).sum()
        loss = reconLoss + klLoss
        return reconLoss, klLoss, loss
    def trainAutoencoder(
        self,
        numEpochs,
        klFactor,
        savedNet,
        learningRate,
        maxAttributeErrorPercent=0.0005,
        sharpAwareMinimization=False,
        sam_rho=0.05,
        Sam_diagnostics=False,
    ):
        opt = torch.optim.AdamW(self.vaeNet.parameters(), learningRate)
        convgHistory = {'reconLoss': [], 'klLoss': [], 'loss': []}
        self.vaeNet.encoder.isTraining = True

        if sharpAwareMinimization and Sam_diagnostics:
            theta_norm0 = self.compute_parameter_norm()
            print(f"[SAM DIAG] Initial ||theta||_2 = {theta_norm0:.6e}")
            print(f"[SAM DIAG] Initial rho/||theta||_2 = {sam_rho / max(theta_norm0, 1e-12):.6e}")

        for epoch in range(numEpochs):
            if not sharpAwareMinimization:
                opt.zero_grad()
                reconLoss, klLoss, loss = self._compute_vae_loss(klFactor)
                loss.backward()
                opt.step()

            else:
                # ---- first pass: compute gradient at current weights ----
                opt.zero_grad()
                reconLoss, klLoss, loss = self._compute_vae_loss(klFactor)
                loss.backward()

                grad_norm_sq = 0.0
                for p in self.vaeNet.parameters():
                    if p.grad is not None:
                        grad_norm_sq += torch.sum(p.grad.detach() ** 2).item()
                grad_norm = (grad_norm_sq ** 0.5) + 1e-12

                # save perturbations and move to adversarial weights
                e_ws = []
                with torch.no_grad():
                    for p in self.vaeNet.parameters():
                        if p.grad is None:
                            e_ws.append(None)
                            continue
                        e_w = sam_rho * p.grad / grad_norm
                        p.add_(e_w)
                        e_ws.append(e_w)

                # ---- second pass: gradient at perturbed weights ----
                opt.zero_grad()
                reconLoss, klLoss, loss = self._compute_vae_loss(klFactor)
                loss.backward()

                # restore original weights
                with torch.no_grad():
                    for p, e_w in zip(self.vaeNet.parameters(), e_ws):
                        if e_w is not None:
                            p.sub_(e_w)

                # optimizer step using gradient from perturbed weights
                opt.step()

            convgHistory['reconLoss'].append(reconLoss.item())
            convgHistory['klLoss'].append((klLoss.item() / klFactor) if klFactor != 0 else 0.0)
            convgHistory['loss'].append(loss.item())

            largestErrorPercent = self.largestEncodingErrorPercent()
            if largestErrorPercent < maxAttributeErrorPercent:
                if sharpAwareMinimization and Sam_diagnostics:
                    theta_norm = self.compute_parameter_norm()
                    print(
                        'Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, '
                        'maxPercentErr {:.3e}, ||theta|| {:.3e}, rho/||theta|| {:.3e}'.format(
                            epoch, reconLoss.item(), klLoss.item(), loss.item(),
                            largestErrorPercent, theta_norm, sam_rho / max(theta_norm, 1e-12)
                        )
                    )
                else:
                    print('Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, maxPercentErr {:.3e}'.format(
                        epoch, reconLoss.item(), klLoss.item(), loss.item(), largestErrorPercent))
                print("Converged!")
                break

            if (epoch % 5000 == 0):
                tag = "SAM" if sharpAwareMinimization else "STD"
                if sharpAwareMinimization and Sam_diagnostics:
                    theta_norm = self.compute_parameter_norm()
                    print(
                        '[{}] Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, '
                        'maxPercentErr {:.3e}, ||theta|| {:.3e}, rho/||theta|| {:.3e}'.format(
                            tag, epoch, reconLoss.item(), klLoss.item(), loss.item(),
                            largestErrorPercent, theta_norm, sam_rho / max(theta_norm, 1e-12)
                        )
                    )
                else:
                    print('[{}] Epoch {:d}: reconLoss {:.3e}, klLoss {:.3e}, loss {:.3e}, maxPercentErr {:.3e}'.format(
                        tag, epoch, reconLoss.item(), klLoss.item(), loss.item(), largestErrorPercent))

        self.vaeNet.encoder.isTraining = False
        torch.save(self.vaeNet.state_dict(), savedNet)
        self.training_latents = self.vaeNet.encoder(self.scaledMaterialData).cpu()
        return convgHistory

    def getMaterialPropertiesAtLatentPoints(self, zPts, compute_gradients=False):
        if compute_gradients:
            zPts_tensor = torch.tensor(zPts, dtype=torch.get_default_dtype()) \
                        if not isinstance(zPts, torch.Tensor) \
                        else zPts.clone().detach()

            def compute_single_element(z_elem):
                z_batch = z_elem.unsqueeze(0)
                decoded = self.vaeNet.decoder(z_batch)
                props = self.getMaterialProperties(decoded)
                return torch.stack([props[name][0] for name in sorted(props.keys())])

            jac_fn = vmap(jacrev(compute_single_element))
            jacobians = jac_fn(zPts_tensor)

            with torch.no_grad():
                decoded = self.vaeNet.decoder(zPts_tensor)
                material_properties = self.getMaterialProperties(decoded)

            gradients = {}
            for idx, prop_name in enumerate(sorted(material_properties.keys())):
                gradients[prop_name] = jacobians[:, idx, :]

            return material_properties, gradients
        else:
            with torch.no_grad():
                decoded = self.vaeNet.decoder(zPts)
                material_properties = self.getMaterialProperties(decoded)
            return material_properties

    def getMaterialProperties(self, decoded):
        properties = {}
        for name, attribute in self.materialAttributes.items():
            idx = attribute['idx']
            scaleMax = attribute['scaleMax']
            scaleMin = attribute['scaleMin']
            if name in ['E0', 'E1', 'E2', 'E3', 'Y0', 'Y1', 'Y2', 'Y3']:
                properties[name] = 10 ** (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
            else:
                properties[name] = (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
        return properties

    def getClosestRealMaterialIndex(self, zDesign):
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return closest_indices

    def getClosestRealMaterialZValues(self, zDesign):
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return zReal[closest_indices].detach().numpy()

    def plotLSR(self, zRealPts, zDesignPts=None, xDesign=None):
        zRealPts = np.asarray(zRealPts, dtype=float)
        if zRealPts.ndim == 1:
            zRealPts = zRealPts.reshape(-1, 1)

        if zDesignPts is not None:
            zDesignPts = np.asarray(zDesignPts, dtype=float)
            if zDesignPts.ndim == 1:
                zDesignPts = zDesignPts.reshape(-1, 1)

        latentDim = zRealPts.shape[1]

        # PCA projection only for latentDim > 3
        if latentDim > 3:
            pca = PCA(n_components=2)
            zRealPts_2d = pca.fit_transform(zRealPts)
            zDesignPts_2d = pca.transform(zDesignPts) if zDesignPts is not None else None

            plt.figure()
            plt.title(f"Latent Space (PCA projection from {latentDim}D)")
            if zDesignPts_2d is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(
                        zDesignPts_2d[mask, 0],
                        zDesignPts_2d[mask, 1],
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.2
                    )
            plt.scatter(zRealPts_2d[:, 0], zRealPts_2d[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=0.4)
            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts_2d[i, 0] + 0.1, zRealPts_2d[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.grid(True)
            plt.legend()
            plt.show()

        # 3D plot
        elif latentDim == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if zDesignPts is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    ax.scatter(
                        zDesignPts[mask, 0],
                        zDesignPts[mask, 1],
                        zDesignPts[mask, 2],
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.6
                    )
            ax.scatter(zRealPts[:, 0], zRealPts[:, 1], zRealPts[:, 2], c='black', marker='*', s=200, label='Real Materials', alpha=0.6)
            for i, label in enumerate(self.materialNames):
                ax.text(zRealPts[i, 0] + 0.1, zRealPts[i, 1], zRealPts[i, 2], str(label), fontsize=10, color='black')
            ax.set_xlabel('$z_1$')
            ax.set_ylabel('$z_2$')
            ax.set_zlabel('$z_3$')
            plt.legend()
            plt.show()

        # 2D plot
        elif latentDim == 2:
            plt.figure()
            if zDesignPts is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(
                        zDesignPts[mask, 0],
                        zDesignPts[mask, 1],
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.2
                    )
            plt.scatter(zRealPts[:, 0], zRealPts[:, 1], c='black', marker='*', s=200, label='Real Materials', alpha=0.4)
            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts[i, 0] + 0.1, zRealPts[i, 1], str(label), fontsize=12, color='black', ha='center', va='bottom')
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.grid(True)
            plt.legend()
            plt.show()

        # 1D plot
        elif latentDim == 1:
            plt.figure(figsize=(8, 4))

            y_real = np.zeros(zRealPts.shape[0])
            plt.scatter(zRealPts[:, 0], y_real, c='black', marker='*', s=200, label='Real Materials', alpha=0.7)

            for i, label in enumerate(self.materialNames):
                plt.text(zRealPts[i, 0] + 0.02, y_real[i] + 0.01, str(label), fontsize=11, color='black')

            if zDesignPts is not None and xDesign is not None:
                mask = np.asarray(xDesign) > 0.5
                if np.any(mask):
                    plt.scatter(
                        zDesignPts[mask, 0],
                        0.02 * np.ones(np.sum(mask)),
                        c='red',
                        marker='o',
                        s=20,
                        label='Optimized Materials',
                        alpha=0.2
                    )

            plt.xlabel('$z$')
            plt.yticks([])
            plt.title('1D Latent Space')
            plt.grid(True)
            plt.legend()
            plt.show()

        else:
            raise ValueError(f"Unsupported latent dimension: {latentDim}")

    def getHeaviestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        heaviest_idx = np.argmax(density_values)
        heaviest_z = z_real[heaviest_idx].detach().cpu().numpy()
        return heaviest_z

    def getLightestMaterial(self):
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties['Density'].detach().cpu().numpy().flatten()
        lightest_idx = np.argmin(density_values)
        lightest_z = z_real[lightest_idx].detach().cpu().numpy()
        return lightest_z
