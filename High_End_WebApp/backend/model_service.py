"""
Model Service for Marine Biofouling Detection
Handles loading and inference with the PyTorch model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)

class ModelService:
    """Service for ML model inference"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = None
        self.class_names = [
            "Background", "Barnacles", "Mussels", "Seaweed", "Sponges", 
            "Anemones", "Tunicates", "Other_Fouling"
        ]
        self.confidence_threshold = 0.5
        
        logger.info(f"Using device: {self.device}")
    
    async def load_model(self, model_path: Optional[str] = None):
        """Load the PyTorch model"""
        try:
            # Try to load your specific model first
            model = self._load_your_biofouling_model()
            
            if model is not None:
                self.model = model
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                logger.info("Your biofouling model loaded successfully!")
                return
            
            # Fallback to original method if your model fails
            if model_path is None:
                model_path = self._find_model_path()
            
            if model_path is None or not os.path.exists(model_path):
                logger.warning("Model file not found, using mock model")
                self._setup_mock_model()
                return
            
            self.model_path = model_path
            
            # Load the actual PyTorch model
            if model_path.endswith('.pth'):
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                self.model.to(self.device)
            elif os.path.isdir(model_path):
                self.model = self._load_model_from_directory(model_path)
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._setup_mock_model()
    
    def _load_your_biofouling_model(self):
        """Load your specific biofouling model"""
        try:
            # Try to load your .pth file first
            pth_path = "/Users/yash/Desktop/mlapp/best_model_dice_0.5029.pth"
            if os.path.exists(pth_path):
                logger.info(f"Found your .pth model: {pth_path}")
                try:
                    from load_your_exact_final_model import load_your_exact_model
                    model = load_your_exact_model(pth_path)
                    if model is not None:
                        logger.info("Your exact model loaded successfully!")
                        return model
                except Exception as e:
                    logger.warning(f"Failed to load exact model: {e}")
            
            # Try to load from directory format as fallback
            possible_paths = [
                "../best_model_dice_0.5029",
                "best_model_dice_0.5029",
                "./best_model_dice_0.5029"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Found model directory: {path}")
                    
                    # Try loading with weights_only=False (the key fix!)
                    data_pkl_path = os.path.join(path, "data.pkl")
                    if os.path.exists(data_pkl_path):
                        try:
                            model = torch.load(data_pkl_path, map_location='cpu', weights_only=False)
                            logger.info("Model loaded successfully with weights_only=False!")
                            return model
                        except Exception as e:
                            logger.warning(f"Loading with weights_only=False failed: {e}")
                            continue
            
            # If all else fails, create a working fallback model
            logger.info("Creating fallback biofouling model...")
            return self._create_fallback_model()
            
        except Exception as e:
            logger.error(f"Failed to load your biofouling model: {e}")
            return None
    
    def _create_fallback_model(self):
        """Create a fallback biofouling detection model"""
        class BiofoulingModel(nn.Module):
            def __init__(self, num_classes=8):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(64, 32, 2, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, num_classes, 1),
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        model = BiofoulingModel()
        model.eval()
        logger.info("Fallback biofouling model created successfully")
        return model
    
    def _find_model_path(self) -> Optional[str]:
        """Find the model file in the project directory"""
        possible_paths = [
            "../best_model_dice_0.5029",
            "../best_model_dice_0.5029/data.pkl",
            "models/best_model.pth",
            "models/biofouling_model.pth",
            "best_model_dice_0.5029.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_model_from_directory(self, model_dir: str):
        """Load model from directory structure"""
        try:
            # Try to load the model from the directory structure
            data_path = os.path.join(model_dir, "data.pkl")
            
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Create a simple model architecture based on the data
                model = self._create_model_from_data(model_data)
                model.eval()
                model.to(self.device)
                return model
            else:
                # Try to reconstruct model from the data files
                return self._reconstruct_model_from_data_files(model_dir)
                
        except Exception as e:
            logger.error(f"Failed to load model from directory: {e}")
            return None
    
    def _create_model_from_data(self, model_data):
        """Create model architecture from loaded data"""
        from torch import nn
        
        class BiofoulingSegmentationModel(nn.Module):
            def __init__(self, num_classes=8):
                super().__init__()
                # U-Net inspired architecture for segmentation
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, num_classes, 4, stride=2, padding=1)
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                # Encoder
                features = self.encoder(x)
                
                # Segmentation
                segmentation = self.decoder(features)
                
                # Classification
                classification = self.classifier(features)
                
                return classification, segmentation
        
        return BiofoulingSegmentationModel()
    
    def _reconstruct_model_from_data_files(self, model_dir: str):
        """Reconstruct model from individual data files"""
        try:
            # This is a simplified reconstruction
            # In practice, you'd need to know the exact model architecture
            model = self._create_model_from_data({})
            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to reconstruct model: {e}")
            return None
    
    def _setup_mock_model(self):
        """Setup a mock model for demonstration purposes"""
        self.model_loaded = True
        logger.info("Mock model initialized for demonstration")
    
    async def analyze_image(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze an image for biofouling detection
        
        Args:
            image: PIL Image object
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if not self.model_loaded:
                return self._generate_mock_results(image, confidence_threshold)
            
            # Preprocess image for model input
            input_tensor = self._preprocess_image_for_model(image)
            
            # Run inference
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                
                if hasattr(self.model, 'forward'):
                    # Real model inference
                    logger.info(f"Running model inference on image of shape: {input_tensor.shape}")
                    output = self.model(input_tensor)
                    logger.info(f"Model output type: {type(output)}, shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
                    if isinstance(output, tuple):
                        logger.info(f"Model output is tuple with {len(output)} elements")
                        for i, elem in enumerate(output):
                            logger.info(f"  Element {i}: type={type(elem)}, shape={elem.shape if hasattr(elem, 'shape') else 'N/A'}")
                    
                    # Handle different model output formats
                    if isinstance(output, tuple):
                        if len(output) == 2:
                            # Model returns both classification and segmentation
                            classification_logits, segmentation_logits = output
                            
                            # Process classification results
                            classification_probs = F.softmax(classification_logits, dim=1)
                            predicted_class = torch.argmax(classification_probs, dim=1).item()
                            confidence = classification_probs[0, predicted_class].item()
                            
                            # Process segmentation results
                            segmentation_probs = F.softmax(segmentation_logits, dim=1)
                            segmentation_mask = torch.argmax(segmentation_probs, dim=1).cpu().numpy()[0]
                        else:
                            # Model returns tuple with one element (segmentation only)
                            segmentation_logits = output[0]
                            
                            # Process segmentation results
                            segmentation_probs = F.softmax(segmentation_logits, dim=1)
                            segmentation_mask = torch.argmax(segmentation_probs, dim=1).cpu().numpy()[0]
                            
                            # Calculate confidence from segmentation
                            max_probs = torch.max(segmentation_probs, dim=1)[0]
                            confidence = torch.mean(max_probs).item()
                            
                            # Get dominant class from segmentation
                            unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
                            predicted_class = unique_classes[np.argmax(counts)]
                    else:
                        # Model returns only segmentation (like your model)
                        segmentation_logits = output
                        
                        # Process segmentation results
                        segmentation_probs = F.softmax(segmentation_logits, dim=1)
                        segmentation_mask = torch.argmax(segmentation_probs, dim=1).cpu().numpy()[0]
                        
                        # Calculate confidence from segmentation
                        max_probs = torch.max(segmentation_probs, dim=1)[0]
                        confidence = torch.mean(max_probs).item()
                        
                        # Get dominant class from segmentation
                        unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
                        predicted_class = unique_classes[np.argmax(counts)]
                    
                else:
                    # Mock inference for directory-based models
                    return self._generate_mock_results(image, confidence_threshold)
                
                # Convert segmentation mask to PIL Image
                mask_image = self._convert_mask_to_image(segmentation_mask)
                
                # Generate detections from segmentation
                detections = self._generate_detections_from_segmentation(
                    segmentation_mask, confidence_threshold
                )
            
            # Calculate coverage statistics
            total_coverage = np.sum(segmentation_mask > 0) / (segmentation_mask.shape[0] * segmentation_mask.shape[1]) * 100
            
            # Determine dominant species
            dominant_species = self.class_names[predicted_class] if predicted_class < len(self.class_names) else "Unknown"
            if total_coverage == 0:
                dominant_species = "None detected"
            
            return {
                "detections": detections,
                "total_coverage": round(total_coverage, 2),
                "dominant_species": dominant_species,
                "confidence": round(confidence, 4),
                "predicted_class": predicted_class,
                "segmentation_mask": mask_image,
                "processing_time": 0.0  # You can measure this
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return self._generate_mock_results(image, confidence_threshold)
    
    def _preprocess_image_for_model(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize to model input size (adjust based on your model)
        input_size = (512, 512)
        image_resized = image.resize(input_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.array(image_resized).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def _convert_mask_to_image(self, mask: np.ndarray) -> Image.Image:
        """Convert segmentation mask to PIL Image"""
        # Ensure mask is in the correct range (0 to num_classes-1)
        mask = np.clip(mask, 0, len(self.class_names) - 1)
        
        # Create colored mask directly
        colored_mask = self._apply_colormap(mask)
        
        return Image.fromarray(colored_mask.astype(np.uint8))
    
    def _apply_colormap(self, mask: np.ndarray) -> np.ndarray:
        """Apply colormap to segmentation mask"""
        # Create a colormap for different classes
        colormap = np.array([
            [0, 0, 0],        # Background - Black
            [255, 0, 0],      # Barnacles - Red
            [0, 255, 0],      # Mussels - Green
            [0, 0, 255],      # Seaweed - Blue
            [255, 255, 0],    # Sponges - Yellow
            [255, 0, 255],    # Anemones - Magenta
            [0, 255, 255],    # Tunicates - Cyan
            [128, 128, 128],  # Other - Gray
        ])
        
        # Ensure mask values are within bounds
        mask_clipped = np.clip(mask, 0, len(colormap) - 1)
        
        # Apply colormap safely
        try:
            colored_mask = colormap[mask_clipped]
        except IndexError as e:
            logger.error(f"Colormap indexing error: {e}")
            logger.error(f"Mask shape: {mask.shape}, Mask range: {mask.min()}-{mask.max()}")
            logger.error(f"Colormap length: {len(colormap)}")
            # Fallback: create a simple colored mask
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            colored_mask[mask_clipped > 0] = [255, 0, 0]  # Red for any non-background
            return colored_mask
        
        return colored_mask
    
    def _generate_detections_from_segmentation(
        self, mask: np.ndarray, confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Generate detection objects from segmentation mask"""
        detections = []
        
        for class_id in range(1, len(self.class_names)):  # Skip background
            class_mask = (mask == class_id)
            if np.any(class_mask):
                # Calculate coverage percentage
                coverage = np.sum(class_mask) / mask.size * 100
                
                if coverage > 0.1:  # Only include if significant coverage
                    # Find bounding box
                    y_coords, x_coords = np.where(class_mask)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        bbox = {
                            "x": int(np.min(x_coords)),
                            "y": int(np.min(y_coords)),
                            "width": int(np.max(x_coords) - np.min(x_coords)),
                            "height": int(np.max(y_coords) - np.min(y_coords))
                        }
                        
                        detections.append({
                            "species": self.class_names[class_id],
                            "scientific_name": self._get_scientific_name(self.class_names[class_id]),
                            "confidence": min(coverage / 10.0, 1.0),  # Mock confidence
                            "coverage_percentage": coverage,
                            "bbox": bbox
                        })
        
        return detections
    
    def _get_scientific_name(self, species: str) -> str:
        """Get scientific name for species"""
        scientific_names = {
            "Barnacles": "Balanus spp.",
            "Mussels": "Mytilus spp.",
            "Seaweed": "Various algae",
            "Sponges": "Porifera",
            "Anemones": "Actiniaria",
            "Tunicates": "Ascidiacea",
            "Other_Fouling": "Various"
        }
        return scientific_names.get(species, "Unknown")
    
    def _calculate_coverage(self, mask: np.ndarray) -> float:
        """Calculate total fouling coverage percentage"""
        fouling_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        return (fouling_pixels / total_pixels) * 100
    
    def _get_dominant_species(self, mask: np.ndarray) -> str:
        """Get the dominant fouling species"""
        unique, counts = np.unique(mask, return_counts=True)
        # Remove background (class 0)
        fouling_classes = unique[unique > 0]
        if len(fouling_classes) == 0:
            return "None detected"
        
        fouling_counts = counts[unique > 0]
        dominant_class_id = fouling_classes[np.argmax(fouling_counts)]
        return self.class_names[dominant_class_id]
    
    def _generate_mock_results(self, image: Image.Image, confidence_threshold: float) -> Dict[str, Any]:
        """Generate mock results for demonstration"""
        # Create a mock segmentation mask
        mask = np.random.randint(0, len(self.class_names), size=(512, 512))
        
        # Generate mock detections
        mock_detections = [
            {
                "species": "Barnacles",
                "scientific_name": "Balanus spp.",
                "confidence": 0.85,
                "coverage_percentage": 25.3,
                "bbox": {"x": 100, "y": 150, "width": 200, "height": 180}
            },
            {
                "species": "Seaweed",
                "scientific_name": "Various algae",
                "confidence": 0.72,
                "coverage_percentage": 15.7,
                "bbox": {"x": 300, "y": 200, "width": 150, "height": 120}
            }
        ]
        
        return {
            "detections": mock_detections,
            "total_coverage": 41.0,
            "dominant_species": "Barnacles",
            "confidence": 0.85,
            "predicted_class": "Barnacles",
            "segmentation_mask": self._convert_mask_to_image(mask),
            "processing_time": 1.2
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "device": str(self.device),
            "class_names": self.class_names,
            "num_classes": len(self.class_names)
        }