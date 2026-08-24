from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
import os

class ProjectController(BaseController):
    def __init__(self):
        super().__init__()

    def get_project_path(self, project_id: str) :

        project_path = os.path.join(self.files_dir, project_id)

        if not os.path.exists(project_path):
            os.makedirs(project_path)
        return project_path  # 👈 سطر الـ return الصريح    
        