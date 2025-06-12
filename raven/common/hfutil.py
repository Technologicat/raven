"""HuggingFace AI model install helper, abstracting a common pattern of conditional install using `huggingface_hub.snapshot_download`."""

__all__ = ["maybe_install_models"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

def maybe_install_models(hf_reponame: str, modelsdir: Optional[str] = None) -> str:
    """Download and install the HuggingFace models into `modelsdir` if the directory does not exist yet. Else do nothing.

    `hf_reponame`: HuggingFace repository to download from, e.g. "OktayAlpk/talking-head-anime-3".
    `modelsdir`: Local path (absolute or relative) to install in. If not specified, install in HF's usual cache location.

    Return the install path.
    """
    # See:
    #   https://huggingface.co/docs/huggingface_hub/en/guides/download

    if modelsdir is not None:  # explicit install location
        logger.info(f"maybe_install_models: Checking for '{hf_reponame}' models at '{modelsdir}'.")
        if os.path.exists(modelsdir):
            logger.info("maybe_install_models: models directory exists. We're good to go!")
        else:
            logger.info(f"maybe_install_models: models not yet installed. Installing from '{hf_reponame}' into '{modelsdir}'. (Don't worry, this will happen only once.)")

            os.makedirs(modelsdir, exist_ok=True)
            return snapshot_download(repo_id=hf_reponame, local_dir=modelsdir)
    else:  # use HF's usual cache location
        logger.info(f"maybe_install_models: Checking for '{hf_reponame}' models at HF's usual cache location.")
        try:
            modelsdir = snapshot_download(repo_id=hf_reponame, local_files_only=True)  # just check if we have it
        except (GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
            logger.info(f"maybe_install_models: Models not yet installed. Installing from '{hf_reponame}' into HF's usual cache location. (Don't worry, this will happen only once.)")
            modelsdir = snapshot_download(repo_id=hf_reponame)
        else:
            logger.info(f"maybe_install_models: Found directory '{modelsdir}'. We're good to go!")
        return modelsdir
