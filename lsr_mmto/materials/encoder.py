"""
encoder.py
==========
MaterialEncoder: a material database read and scaled to [0, 1], the variational autoencoder (networks.py)
trained on it or read from its .nt file, and what the run and the app ask of it: each material's latent point,
the properties decoded at latent points, the closest database material, the network's errors.
"""
from typing import TYPE_CHECKING, Any, Callable

from lsr_mmto.materials.networks import VariationalAutoencoder
import torch
from torch.func import vmap, jacrev
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lsr_mmto.materials.pca import PCA
import matplotlib.cm as cm
import os
import sys
import warnings
from lsr_mmto.materials.databases import (SHOWN, once_per_file_version, attribute_name, database_pca, log_scaled,
                                          normalize)

if TYPE_CHECKING:
    from lsr_mmto.materials.databases import VAEParams

STOP_CHECK_EVERY = 100            # epochs between two asks of trainAutoencoder's should_stop


class TrainingStopped(Exception):
    """The training was asked to stop (trainAutoencoder's should_stop); nothing was saved."""

    def __init__(self, epoch: int) -> None:
        super().__init__(f"the material network's training was stopped at epoch {epoch}; nothing was saved")
        self.epoch = epoch


class NetworkRunCancelled(Exception):
    """The material network reproduces the database less closely than the project's warning limit, and the
    run was not wanted (_run_topopt's on_network_error answered False): nothing was optimized."""


def network_warning(errors: dict[str, float], limit: float) -> str | None:
    """The warning for a network whose largest error (errors: {attribute: % of its range}) is above limit (%),
    listing every attribute, largest first; None when it is within the limit."""
    if not errors or max(errors.values()) <= limit:
        return None
    listed = ", ".join(f"{SHOWN.get(k, k)} {v:.1f}%" for k, v in sorted(errors.items(), key=lambda kv: -kv[1]))
    return (f"the material network reproduces the database less closely than the {limit:g}% warning limit: "
            f"{listed} (each the largest error over the materials, in % of the property's range)")


@once_per_file_version
def network_reconstruction(xlsx: str, nt: str) -> list[dict[str, Any]] | None:
    """MaterialEncoder.reconstruction() of the network saved in nt for the database xlsx, its shape read
    from the file; None when nt is not a network for this database."""
    from lsr_mmto.materials.databases import VAEParams
    from lsr_mmto.problem.project import network_shape
    shape = network_shape(nt)
    if shape is None:
        return None
    params = VAEParams()
    params.vae_hiddenDim, params.latentDim = shape[1], shape[2]
    enc = MaterialEncoder(params)
    enc.readExcel(xlsx)
    if enc.nAttributes != shape[0]:
        return None
    enc.loadAutoencoderFromFile(nt)
    return enc.reconstruction()


def largest_errors(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float, str | None]]:
    """{property: (largest error in % of its range, largest difference in its unit, unit)} over the
    materials, from MaterialEncoder.reconstruction() rows."""
    out = {}
    for r in rows:
        pct, diff, _ = out.get(r["property"], (0.0, 0.0, r["unit"]))
        out[r["property"]] = (max(pct, r["percent_of_range"]), max(diff, abs(r["difference"])), r["unit"])
    return out


class MaterialEncoder:
    """A material database and its network. readExcel reads the database (and makes an untrained network of
    the shape vae_params gives); loadAutoencoderFromFile or trainAutoencoder then gives the network its
    weights. Latent points are rows, (points, latentDim); properties are {database column: (points,)}."""

    nAttributes: int                         # property columns of the database (0 before readExcel)
    vae_params: "VAEParams"
    excel_file: str
    nMaterials: int
    vaeSettings: dict[str, dict[str, int]]
    vaeNet: VariationalAutoencoder
    rawData: np.ndarray                      # the database's numbers, (materials, attributes)
    materialNames: list[Any]                 # as pandas reads the first column: text, or a number
    scaledMaterialData: torch.Tensor         # rawData scaled to [0, 1] (normalize), (materials, attributes)
    materialAttributes: dict[str, dict[str, Any]]   # {column: {"idx", "unit", "scaleMin", "scaleMax"}}
                                                    # (unit None when its cell is blank)
    attribute_means: np.ndarray              # of the scaled columns
    attribute_stds: np.ndarray
    intrinsic_dim: float                     # d_eff, the participation ratio (databases.database_pca)
    pca_explained_variance: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    training_latents: torch.Tensor           # each material's latent point, (materials, latentDim)

    def __init__(self, vae_params: "VAEParams") -> None:
        self.nAttributes = 0
        self.vae_params = vae_params

    def readExcel(self, excel_file: str) -> None:
        """Read and scale the database (preprocessData) and make an untrained network of its shape."""
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

    def preprocessData(self) -> None:
        """Read excel_file: row 1 the column names (aliases made registry names), row 2 their units (None when
        blank, as databases.database_table reads them), then one row per material, its name first. Sets rawData, materialNames, scaledMaterialData, materialAttributes,
        attribute_means, attribute_stds, and the database's PCA (computeParticipationRatio)."""
        with warnings.catch_warnings():
            # the databases carry Excel data validation, which openpyxl drops
            warnings.filterwarnings("ignore", message="Data Validation extension")
            df = pd.read_excel(self.excel_file, header=None)
        attribute_names = [attribute_name(n) for n in df.iloc[0, 1:].tolist()]   # aliases -> registry names
        units = [None if (u is None or str(u).strip() in ("", "nan")) else str(u).strip() for u in df.iloc[1, 1:]]
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

    def computeParticipationRatio(self) -> float:
        """
        Effective intrinsic dimensionality estimate from PCA eigenvalues of the normalized database:
        databases.database_pca, worked out once per version of the file. Shown by results/tables.pca and the
        Materials window, not printed.
        """
        self.intrinsic_dim, self.pca_explained_variance, self.pca_explained_variance_ratio = \
            database_pca(self.excel_file)
        return self.intrinsic_dim


    def largestEncodingErrorPercent(self) -> float:
        """The largest error over every material and attribute, encoded and decoded, in % of its range (of its
        log10 for a log-scaled attribute)."""
        was = self.vaeNet.encoder.isTraining                # put back as found: left on, the encoder samples at random
        self.vaeNet.encoder.isTraining = False
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            diffMatrix = self.scaledMaterialData - decoded
            largestErrorPercent = 100.0 * torch.max(torch.abs(diffMatrix)).item()
        self.vaeNet.encoder.isTraining = was
        return largestErrorPercent

    def encodingErrorsPercent(self) -> dict[str, float]:
        """{attribute: the largest error over the materials, in % of the attribute's range}."""
        was = self.vaeNet.encoder.isTraining
        self.vaeNet.encoder.isTraining = False
        with torch.no_grad():
            diff = self.scaledMaterialData - self.vaeNet.decoder(self.vaeNet.encoder(self.scaledMaterialData))
        self.vaeNet.encoder.isTraining = was
        return {name: 100.0 * float(diff[:, a["idx"]].abs().max()) for name, a in self.materialAttributes.items()}

    def reconstruction(self) -> list[dict[str, Any]]:
        """Every material's every property as the database gives it and as the network gives it back
        (encoded to its latent point, then decoded), one dict each: material, property, unit, database,
        network, difference (network - database, in the unit), percent_of_range (the difference in % of
        the property's range over the database; for a log-scaled attribute, of the range of its log10)."""
        was = self.vaeNet.encoder.isTraining                # put back as found: left on, the encoder samples at random
        self.vaeNet.encoder.isTraining = False
        with torch.no_grad():
            decoded = self.vaeNet.decoder(self.vaeNet.encoder(self.scaledMaterialData))
            diff = (self.scaledMaterialData - decoded).abs()
            props = self.getMaterialProperties(decoded)
        self.vaeNet.encoder.isTraining = was
        rows = []
        for i, material in enumerate(self.materialNames):
            for name, a in self.materialAttributes.items():
                database, network = float(self.rawData[i, a["idx"]]), float(props[name][i])
                rows.append({"material": str(material), "property": name, "unit": a["unit"],
                             "database": database, "network": network, "difference": network - database,
                             "percent_of_range": 100.0 * float(diff[i, a["idx"]])})
        return rows

    def printEncodingErrors(self) -> None:
        """Print each property's largest error (largest_errors of reconstruction()) as a table."""
        print("-" * 72)
        print("{:<25} {:>22} {:>23}".format("Property", "Largest error", "Largest difference"))
        print("{:<25} {:>22} {:>23}".format("", "[% of its range]", "[its unit]"))
        print("-" * 72)
        for name, (pct, diff, unit) in largest_errors(self.reconstruction()).items():
            print("{:<25} {:>22.6f} {:>23}".format(SHOWN.get(name, name), pct, f"{diff:.6g} {unit or ''}".strip()))
        print("-" * 72)
    def compute_parameter_norm(self) -> float:
        """The 2-norm of every parameter of the network (weights and biases) together."""
        total_sq = 0.0
        with torch.no_grad():
            for p in self.vaeNet.parameters():
                total_sq += torch.sum(p.detach() ** 2).item()
        return total_sq ** 0.5
    def loadAutoencoderFromFile(self, fileName: str) -> None:
        """The network's weights read from fileName (a .nt file); sets training_latents."""
        self.vaeNet.load_state_dict(torch.load(fileName))
        self.vaeNet.eval()
        self.training_latents = self.vaeNet.encoder(self.scaledMaterialData).cpu()
    def _compute_vae_loss(self, klFactor: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(reconstruction loss, klFactor * KL, their sum) of the whole database, keeping the autograd graph."""
        predData = self.vaeNet(self.scaledMaterialData)
        klLoss = klFactor * self.vaeNet.encoder.kl
        reconLoss = ((self.scaledMaterialData - predData) ** 2).sum()
        loss = reconLoss + klLoss
        return reconLoss, klLoss, loss
    def trainAutoencoder(
        self,
        numEpochs: int,
        klFactor: float,
        savedNet: str,
        learningRate: float,
        maxAttributeErrorPercent: float = 0.0005,
        sharpAwareMinimization: bool = False,
        sam_rho: float = 0.05,
        Sam_diagnostics: bool = False,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, list[float]]:
        """Train the network on the database, save its weights to savedNet and set
        training_latents. It trains for up to numEpochs (AdamW), stopping once the largest error is below
        maxAttributeErrorPercent; a largest error still at or above it is printed, not refused. Returns the loss
        history."""
        # should_stop: optional function, asked every STOP_CHECK_EVERY epochs; True ends the training by
        # raising TrainingStopped, before the network is saved (savedNet is left as it was).
        opt = torch.optim.AdamW(self.vaeNet.parameters(), learningRate)
        convgHistory = {'reconLoss': [], 'klLoss': [], 'loss': []}
        self.vaeNet.encoder.isTraining = True

        if sharpAwareMinimization and Sam_diagnostics:
            theta_norm0 = self.compute_parameter_norm()
            print(f"[SAM DIAG] Initial ||theta||_2 = {theta_norm0:.6e}")
            print(f"[SAM DIAG] Initial rho/||theta||_2 = {sam_rho / max(theta_norm0, 1e-12):.6e}")

        for epoch in range(numEpochs):
            if should_stop is not None and epoch % STOP_CHECK_EVERY == 0 and should_stop():
                raise TrainingStopped(epoch)
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
        error = self.largestEncodingErrorPercent()
        if error >= maxAttributeErrorPercent:
            print(f"The network's largest error is {error:.2f}% (asked {maxAttributeErrorPercent:g}%).")
        torch.save(self.vaeNet.state_dict(), savedNet)
        self.training_latents = self.vaeNet.encoder(self.scaledMaterialData).cpu()
        return convgHistory

    def getMaterialPropertiesAtLatentPoints(self, zPts: torch.Tensor, compute_gradients: bool = False) \
            -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """The properties at latent points zPts, without the autograd graph; with compute_gradients, also each
        property's gradient w.r.t. its point, {column: (points, latentDim)}. Not used anywhere in this repository."""
        if compute_gradients:
            zPts_tensor = torch.tensor(zPts, dtype=torch.get_default_dtype()) \
                        if not isinstance(zPts, torch.Tensor) \
                        else zPts.clone().detach()

            def compute_single_element(z_elem: torch.Tensor) -> torch.Tensor:
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

    def getMaterialProperties(self, decoded: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decoder outputs (points, attributes) in [0, 1] back to each column's numbers, {column: (points,)}:
        normalize undone (and log10 undone for a log-scaled column); the autograd graph is kept when decoded has one."""
        properties = {}
        for name, attribute in self.materialAttributes.items():
            idx = attribute['idx']
            scaleMax = attribute['scaleMax']
            scaleMin = attribute['scaleMin']
            if log_scaled(name):                            # scaled as log10 by normalize: undone the same way
                properties[name] = 10 ** (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
            else:
                properties[name] = (decoded[:, idx] * (scaleMax - scaleMin) + scaleMin)
        return properties

    def getClosestRealMaterialIndex(self, zDesign: torch.Tensor) -> torch.Tensor:
        """The database row of the material whose latent point is closest to each of zDesign, (points,)."""
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return closest_indices

    def getClosestRealMaterialZValues(self, zDesign: torch.Tensor) -> np.ndarray:
        """The latent point of the closest database material to each of zDesign, (points, latentDim)."""
        with torch.no_grad():
            zReal = self.vaeNet.encoder(self.scaledMaterialData)

        distances = torch.cdist(zDesign, zReal)
        closest_indices = torch.argmin(distances, dim=1)
        return zReal[closest_indices].detach().numpy()

    def plotLSR(self, zRealPts: np.ndarray, zDesignPts: np.ndarray | None = None,
                xDesign: np.ndarray | None = None) -> None:
        """A matplotlib window of the latent space: the materials, and the design's points where xDesign > 0.5
        (none without xDesign); projected to 2D by PCA above 3 dimensions. Not used anywhere in this repository."""
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

    def getHeaviestMaterial(self) -> np.ndarray:
        """The latent point (latentDim,) of the material the network gives the highest density."""
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties[_density_column(decoded_properties)].detach().cpu().numpy().flatten()
        heaviest_idx = np.argmax(density_values)
        heaviest_z = z_real[heaviest_idx].detach().cpu().numpy()
        return heaviest_z

    def getLightestMaterial(self) -> np.ndarray:
        """The latent point (latentDim,) of the material the network gives the lowest density."""
        with torch.no_grad():
            z_real = self.vaeNet.encoder(self.scaledMaterialData)
            decoded = self.vaeNet.decoder(z_real)
            decoded_properties = self.getMaterialProperties(decoded)

        density_values = decoded_properties[_density_column(decoded_properties)].detach().cpu().numpy().flatten()
        lightest_idx = np.argmin(density_values)
        lightest_z = z_real[lightest_idx].detach().cpu().numpy()
        return lightest_z


def _density_column(properties: dict[str, torch.Tensor]) -> str:
    """The column the lightest and heaviest starts compare: the first one that measures density."""
    from lsr_mmto.materials.databases import density_column
    name = density_column(properties)
    if name is not None:
        return name
    raise ValueError("starting from the lightest or the heaviest material needs a database column that "
                     "measures density")
