from django.apps import AppConfig


class MedscanConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'medScan'
    
    def ready(self):
        from . import model_loader
        model_loader.load_model_once()

