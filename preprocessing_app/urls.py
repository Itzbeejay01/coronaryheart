from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('processing/<int:image_id>/', views.processing_status_view, name='processing_status'),
    path('progress/<int:image_id>/', views.get_processing_progress, name='get_progress'),
    path('result/<int:image_id>/', views.result, name='result'),
    path('batch-status/<uuid:batch_id>/', views.batch_status, name='batch_status'),
    path('get_processing_progress/<int:image_id>/', views.get_processing_progress, name='get_processing_progress'),
    path('predict-diagnosis/', views.predict_diagnosis_view, name='predict_diagnosis'),
    path('predict-diagnosis/result/', views.predict_diagnosis_result_view, name='predict_diagnosis_result'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)