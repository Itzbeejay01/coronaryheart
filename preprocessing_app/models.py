from django.db import models
import os
import uuid
from django.utils import timezone

# Create your models here.

class BaseImage(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error')
    ]
    
    title = models.CharField(max_length=200)
    upload_date = models.DateTimeField(auto_now_add=True)
    processed_date = models.DateTimeField(null=True, blank=True)
    image_dimensions = models.CharField(max_length=50, null=True, blank=True)
    preprocessing_steps = models.JSONField(default=dict)
    processed_image = models.ImageField(upload_to='processed/', null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    class Meta:
        abstract = True
    
    def __str__(self):
        return self.title
    
    def get_processed_filename(self):
        return os.path.basename(self.processed_image.name) if self.processed_image else None

class DicomImage(BaseImage):
    dicom_file = models.FileField(upload_to='uploads/dicom/')
    visualization_image = models.ImageField(upload_to='uploads/dicom/preview/', null=True, blank=True)
    dicom_info = models.JSONField(default=dict)
    
    def filename(self):
        return os.path.basename(self.dicom_file.name)

class JpegImage(BaseImage):
    IMAGE_TYPE_CHOICES = [
        ('jpeg', 'JPEG Coronary Image'),
        ('png', 'PNG Coronary Image'),
        ('other', 'Other Coronary Image')
    ]
    
    image_file = models.ImageField(upload_to='uploads/jpeg/')
    visualization_image = models.ImageField(upload_to='uploads/jpeg/preview/', null=True, blank=True)
    image_type = models.CharField(max_length=20, choices=IMAGE_TYPE_CHOICES, default='jpeg')
    metadata = models.JSONField(default=dict)
    
    def filename(self):
        return os.path.basename(self.image_file.name)

class ProcessingBatch(models.Model):
    batch_id = models.UUIDField(default=uuid.uuid4, primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed = models.BooleanField(default=False)
    total_files = models.IntegerField(default=0)
    processed_files = models.IntegerField(default=0)
    dicom_images = models.ManyToManyField(DicomImage, blank=True)
    jpeg_images = models.ManyToManyField(JpegImage, blank=True)

class ProcessedImage(models.Model):
    original_image = models.ImageField(upload_to='original_images/')
    processed_image = models.ImageField(upload_to='processed_images/')
    original_dimensions = models.CharField(max_length=50)
    processed_dimensions = models.CharField(max_length=50)
    psnr = models.FloatField()
    mse = models.FloatField()
    ssim = models.FloatField()
    uqi = models.FloatField()
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Processed Image {self.id} - {self.created_at}"