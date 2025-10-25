"""
Hugging Face Hub integration for BharatFM models
"""

import os
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import shutil

try:
    from huggingface_hub import HfApi, HfFolder, Repository
    from huggingface_hub.utils import RepositoryNotFoundError
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    raise ImportError("huggingface_hub is required for hub integration")


class BharatHubManager:
    """Hugging Face Hub manager for BharatFM models"""
    
    def __init__(
        self,
        token: str = None,
        organization: str = None,
        default_repo: str = "bharat-ai"
    ):
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub is required for hub integration")
            
        self.token = token or os.environ.get("HF_TOKEN")
        self.organization = organization
        self.default_repo = default_repo
        
        self.api = HfApi(token=self.token)
        self.logger = logging.getLogger(__name__)
        
        # Login if token provided
        if self.token:
            self.login()
            
    def login(self):
        """Login to Hugging Face Hub"""
        HfFolder.save_token(self.token)
        self.logger.info("Logged in to Hugging Face Hub")
        
    def logout(self):
        """Logout from Hugging Face Hub"""
        HfFolder.delete_token()
        self.logger.info("Logged out from Hugging Face Hub")
        
    def create_repository(
        self,
        repo_name: str,
        organization: str = None,
        private: bool = False,
        exist_ok: bool = False
    ) -> str:
        """Create a new repository on Hugging Face Hub"""
        if organization is None:
            organization = self.organization
            
        if organization:
            repo_id = f"{organization}/{repo_name}"
        else:
            repo_id = repo_name
            
        try:
            repo_url = self.api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=exist_ok
            )
            self.logger.info(f"Created repository: {repo_id}")
            return repo_id
            
        except Exception as e:
            self.logger.error(f"Error creating repository: {e}")
            raise
            
    def delete_repository(self, repo_id: str):
        """Delete a repository from Hugging Face Hub"""
        try:
            self.api.delete_repo(repo_id=repo_id)
            self.logger.info(f"Deleted repository: {repo_id}")
            
        except Exception as e:
            self.logger.error(f"Error deleting repository: {e}")
            raise
            
    def upload_model(
        self,
        model_path: str,
        repo_id: str,
        commit_message: str = None,
        commit_description: str = None,
        create_repo: bool = False,
        private: bool = False
    ) -> str:
        """Upload model to Hugging Face Hub"""
        if commit_message is None:
            commit_message = f"Upload model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        try:
            # Create repository if needed
            if create_repo:
                self.create_repository(
                    repo_name=repo_id.split("/")[-1],
                    organization=repo_id.split("/")[0] if "/" in repo_id else None,
                    private=private,
                    exist_ok=True
                )
                
            # Upload files
            repo_url = self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                commit_message=commit_message,
                commit_description=commit_description
            )
            
            self.logger.info(f"Uploaded model to: {repo_id}")
            return repo_url
            
        except Exception as e:
            self.logger.error(f"Error uploading model: {e}")
            raise
            
    def upload_file(
        self,
        file_path: str,
        repo_id: str,
        path_in_repo: str = None,
        commit_message: str = None,
        repo_type: str = None
    ) -> str:
        """Upload a single file to Hugging Face Hub"""
        if path_in_repo is None:
            path_in_repo = os.path.basename(file_path)
            
        if commit_message is None:
            commit_message = f"Upload {path_in_repo} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        try:
            self.api.upload_file(
                path_or_fileobj=file_path,
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
                repo_type=repo_type
            )
            
            self.logger.info(f"Uploaded file to: {repo_id}/{path_in_repo}")
            return f"{repo_id}/{path_in_repo}"
            
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            raise
            
    def download_model(
        self,
        repo_id: str,
        local_path: str = None,
        revision: str = None,
        repo_type: str = None
    ) -> str:
        """Download model from Hugging Face Hub"""
        if local_path is None:
            local_path = tempfile.mkdtemp()
            
        try:
            self.api.hf_hub_download(
                repo_id=repo_id,
                filename="config.json",  # Just to check if repo exists
                local_dir=local_path,
                revision=revision,
                repo_type=repo_type
            )
            
            # Download all files
            repo_url = self.api.snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                revision=revision,
                repo_type=repo_type
            )
            
            self.logger.info(f"Downloaded model from: {repo_id}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")
            raise
            
    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_path: str = None,
        revision: str = None,
        repo_type: str = None
    ) -> str:
        """Download a single file from Hugging Face Hub"""
        try:
            file_path = self.api.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_path,
                revision=revision,
                repo_type=repo_type
            )
            
            self.logger.info(f"Downloaded file from: {repo_id}/{filename}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            raise
            
    def list_repo_files(
        self,
        repo_id: str,
        revision: str = None,
        repo_type: str = None
    ) -> List[str]:
        """List files in a repository"""
        try:
            files = self.api.list_repo_files(
                repo_id=repo_id,
                revision=revision,
                repo_type=repo_type
            )
            
            return files
            
        except Exception as e:
            self.logger.error(f"Error listing repository files: {e}")
            raise
            
    def delete_file(
        self,
        repo_id: str,
        path_in_repo: str,
        commit_message: str = None,
        repo_type: str = None
    ):
        """Delete a file from repository"""
        if commit_message is None:
            commit_message = f"Delete {path_in_repo} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        try:
            self.api.delete_file(
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
                repo_type=repo_type
            )
            
            self.logger.info(f"Deleted file: {repo_id}/{path_in_repo}")
            
        except Exception as e:
            self.logger.error(f"Error deleting file: {e}")
            raise
            
    def list_repositories(
        self,
        organization: str = None,
        sort: str = "createdAt",
        direction: str = "desc",
        limit: int = None
    ) -> List[Dict]:
        """List repositories"""
        try:
            repos = self.api.list_repos(
                organization=organization,
                sort=sort,
                direction=direction,
                limit=limit
            )
            
            return [repo.to_dict() for repo in repos]
            
        except Exception as e:
            self.logger.error(f"Error listing repositories: {e}")
            raise
            
    def get_repository_info(self, repo_id: str) -> Dict:
        """Get repository information"""
        try:
            repo = self.api.repo_info(repo_id)
            return repo.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error getting repository info: {e}")
            raise
            
    def update_repository_info(
        self,
        repo_id: str,
        private: bool = None,
        description: str = None,
        default_branch: str = None
    ):
        """Update repository information"""
        try:
            self.api.update_repo_info(
                repo_id=repo_id,
                private=private,
                description=description,
                default_branch=default_branch
            )
            
            self.logger.info(f"Updated repository info: {repo_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating repository info: {e}")
            raise
            
    def create_model_card(
        self,
        repo_id: str,
        model_name: str,
        model_description: str,
        model_type: str = "glm",
        languages: List[str] = None,
        tags: List[str] = None,
        license: str = "apache-2.0",
        datasets: List[str] = None,
        metrics: List[str] = None
    ):
        """Create and upload model card"""
        if languages is None:
            languages = ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa"]
            
        if tags is None:
            tags = ["bharatfm", "indian-languages", "foundation-model"]
            
        model_card = f"""---
language: {', '.join(languages)}
license: {license}
tags: {', '.join(tags)}
datasets: {datasets if datasets else '[]'}
metrics: {metrics if metrics else '[]'}
---

# {model_name}

{model_description}

## Model Details

- **Model Type:** {model_type}
- **Languages:** {', '.join(languages)}
- **License:** {license}
- **Framework:** BharatFM

## Usage

```python
from bharatfm import BharatModel

model = BharatModel.from_pretrained("{repo_id}")
```

## Training

This model was trained as part of the Bharat Foundation Model Framework.

## Limitations

This model is designed specifically for Indian languages and may not perform well on other language families.

## Ethical Considerations

This model should be used responsibly and in accordance with the license terms.
"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(model_card)
            temp_path = f.name
            
        try:
            # Upload model card
            self.upload_file(
                file_path=temp_path,
                repo_id=repo_id,
                path_in_repo="README.md",
                commit_message="Add model card"
            )
            
            self.logger.info(f"Created model card for: {repo_id}")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    def create_model_config(
        self,
        repo_id: str,
        config: Dict[str, Any]
    ):
        """Create and upload model configuration"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            temp_path = f.name
            
        try:
            # Upload config
            self.upload_file(
                file_path=temp_path,
                repo_id=repo_id,
                path_in_repo="config.json",
                commit_message="Add model configuration"
            )
            
            self.logger.info(f"Created model config for: {repo_id}")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    def clone_repository(
        self,
        repo_id: str,
        local_path: str = None,
        revision: str = None
    ) -> str:
        """Clone repository to local path"""
        if local_path is None:
            local_path = tempfile.mkdtemp()
            
        try:
            repo = Repository(
                local_dir=local_path,
                clone_from=repo_id,
                revision=revision
            )
            
            self.logger.info(f"Cloned repository: {repo_id} to {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Error cloning repository: {e}")
            raise
            
    def push_to_hub(
        self,
        local_path: str,
        repo_id: str,
        commit_message: str = None,
        commit_description: str = None
    ):
        """Push local repository to hub"""
        if commit_message is None:
            commit_message = f"Push changes - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        try:
            repo = Repository(local_dir=local_path)
            repo.push_to_hub(
                commit_message=commit_message,
                commit_description=commit_description
            )
            
            self.logger.info(f"Pushed changes to: {repo_id}")
            
        except Exception as e:
            self.logger.error(f"Error pushing to hub: {e}")
            raise
            
    def create_dataset(
        self,
        dataset_path: str,
        repo_id: str,
        private: bool = False,
        commit_message: str = None
    ) -> str:
        """Create and upload dataset"""
        if commit_message is None:
            commit_message = f"Upload dataset - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
        try:
            repo_url = self.api.upload_folder(
                folder_path=dataset_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message
            )
            
            self.logger.info(f"Created dataset: {repo_id}")
            return repo_url
            
        except Exception as e:
            self.logger.error(f"Error creating dataset: {e}")
            raise
            
    def list_models(
        self,
        author: str = None,
        filter: str = None,
        sort: str = "downloads",
        direction: str = "desc",
        limit: int = None
    ) -> List[Dict]:
        """List models on the hub"""
        try:
            models = self.api.list_models(
                author=author,
                filter=filter,
                sort=sort,
                direction=direction,
                limit=limit
            )
            
            return [model.to_dict() for model in models]
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            raise
            
    def whoami(self) -> Dict:
        """Get current user information"""
        try:
            user_info = self.api.whoami()
            return user_info
            
        except Exception as e:
            self.logger.error(f"Error getting user info: {e}")
            raise


class ModelHubManager:
    """High-level model hub management interface"""
    
    def __init__(self, hub_manager: BharatHubManager):
        self.hub_manager = hub_manager
        self.logger = logging.getLogger(__name__)
        
    def publish_model(
        self,
        model_path: str,
        model_name: str,
        model_description: str,
        model_type: str = "glm",
        config: Dict[str, Any] = None,
        languages: List[str] = None,
        tags: List[str] = None,
        license: str = "apache-2.0",
        private: bool = False,
        organization: str = None
    ) -> str:
        """Publish a complete model to the hub"""
        
        # Determine repo ID
        if organization:
            repo_id = f"{organization}/{model_name}"
        else:
            repo_id = model_name
            
        # Create repository
        self.hub_manager.create_repository(
            repo_name=model_name,
            organization=organization,
            private=private,
            exist_ok=True
        )
        
        # Upload model files
        self.hub_manager.upload_model(
            model_path=model_path,
            repo_id=repo_id,
            commit_message=f"Publish {model_name} model"
        )
        
        # Create model card
        self.hub_manager.create_model_card(
            repo_id=repo_id,
            model_name=model_name,
            model_description=model_description,
            model_type=model_type,
            languages=languages,
            tags=tags,
            license=license
        )
        
        # Create model config
        if config:
            self.hub_manager.create_model_config(repo_id, config)
            
        self.logger.info(f"Published model: {repo_id}")
        return repo_id
        
    def download_model_package(
        self,
        repo_id: str,
        local_path: str = None,
        revision: str = None
    ) -> str:
        """Download complete model package"""
        return self.hub_manager.download_model(
            repo_id=repo_id,
            local_path=local_path,
            revision=revision
        )
        
    def update_model(
        self,
        repo_id: str,
        model_path: str,
        commit_message: str = None,
        config: Dict[str, Any] = None
    ):
        """Update existing model on hub"""
        
        # Upload updated model files
        self.hub_manager.upload_model(
            model_path=model_path,
            repo_id=repo_id,
            commit_message=commit_message
        )
        
        # Update config if provided
        if config:
            self.hub_manager.create_model_config(repo_id, config)
            
        self.logger.info(f"Updated model: {repo_id}")
        
    def list_my_models(self) -> List[Dict]:
        """List models owned by current user"""
        user_info = self.hub_manager.whoami()
        return self.hub_manager.list_models(author=user_info["name"])
        
    def search_models(
        self,
        query: str,
        author: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search for models on the hub"""
        return self.hub_manager.list_models(
            author=author,
            filter=query,
            limit=limit
        )
        
    def get_model_info(self, repo_id: str) -> Dict:
        """Get detailed model information"""
        try:
            # Get repository info
            repo_info = self.hub_manager.get_repository_info(repo_id)
            
            # List files
            files = self.hub_manager.list_repo_files(repo_id)
            
            # Try to get config
            config = None
            if "config.json" in files:
                try:
                    config_path = self.hub_manager.download_file(
                        repo_id=repo_id,
                        filename="config.json"
                    )
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    os.unlink(config_path)
                except:
                    pass
                    
            return {
                "repo_info": repo_info,
                "files": files,
                "config": config
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            raise