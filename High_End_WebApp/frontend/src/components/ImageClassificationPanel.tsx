import React, { useState } from 'react';
import { Upload, Brain, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import axios from 'axios';

interface ClassificationResult {
  label: string;
  score: number;
  confidence: string;
  is_marine_related: boolean;
  category: string;
  relevance: string;
}

interface ApiClassificationResult {
  success: boolean;
  model_used: string;
  classifications: ClassificationResult[];
  note?: string;
}

interface CombinedAnalysis {
  marine_items_detected: any[];
  total_marine_items: number;
  average_confidence: number;
  confidence_level: string;
}

interface ClassificationSummary {
  primary_fouling_type: string;
  fouling_categories: Record<string, number>;
  total_items: number;
  confidence_level: string;
  recommendation: string;
}

interface MultiModelResults {
  individual_results: Record<string, ApiClassificationResult>;
  combined_analysis: CombinedAnalysis;
  summary: ClassificationSummary;
}

const ImageClassificationPanel: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [singleModelResults, setSingleModelResults] = useState<ApiClassificationResult | null>(null);
  const [multiModelResults, setMultiModelResults] = useState<MultiModelResults | null>(null);
  const [selectedModel, setSelectedModel] = useState('resnet50');

  const models = [
    { value: 'resnet50', label: 'ResNet-50 (Microsoft)' },
    { value: 'vit', label: 'Vision Transformer (Google)' },
    { value: 'convnext', label: 'ConvNeXt (Facebook)' }
  ];

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
      setSingleModelResults(null);
      setMultiModelResults(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    multiple: false
  });

  const classifySingleModel = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model', selectedModel);

      const response = await axios.post('/api/classify-image-single', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setSingleModelResults(response.data.classification_results);
      toast.success('Single model classification completed!');
    } catch (error) {
      console.error('Classification failed:', error);
      toast.error('Classification failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const classifyMultiModel = async () => {
    if (!selectedFile) {
      toast.error('Please select an image first');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('/api/classify-image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setMultiModelResults(response.data.classification_results);
      toast.success('Multi-model classification completed!');
    } catch (error) {
      console.error('Multi-model classification failed:', error);
      toast.error('Multi-model classification failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderClassificationResults = () => {
    if (singleModelResults) {
      return (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <Brain className="w-5 h-5 mr-2 text-blue-600" />
            Single Model Results ({singleModelResults.model_used})
          </h3>
          
          {singleModelResults.note && (
            <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-800">{singleModelResults.note}</p>
            </div>
          )}

          <div className="space-y-3">
            {singleModelResults.classifications.slice(0, 10).map((item, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  item.is_marine_related
                    ? 'bg-blue-50 border-blue-200'
                    : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{item.label}</h4>
                    <p className="text-sm text-gray-600">{item.category}</p>
                  </div>
                  <div className="text-right">
                    <span className="text-lg font-semibold text-blue-600">
                      {item.confidence}
                    </span>
                    <p className="text-xs text-gray-500">
                      {item.relevance} relevance
                    </p>
                  </div>
                </div>
                <div className="mt-2">
                  <div className="flex items-center space-x-2">
                    {item.is_marine_related ? (
                      <CheckCircle className="w-4 h-4 text-blue-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-gray-400" />
                    )}
                    <span className="text-xs text-gray-600">
                      {item.is_marine_related ? 'Marine-related' : 'General classification'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      );
    }

    if (multiModelResults) {
      return (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <Brain className="w-5 h-5 mr-2 text-green-600" />
            Multi-Model Combined Analysis
          </h3>

          {/* Summary */}
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <h4 className="font-semibold text-green-800 mb-2">Analysis Summary</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-green-600 font-medium">Primary Fouling</p>
                <p className="text-green-800">{multiModelResults.summary.primary_fouling_type}</p>
              </div>
              <div>
                <p className="text-green-600 font-medium">Total Items</p>
                <p className="text-green-800">{multiModelResults.summary.total_items}</p>
              </div>
              <div>
                <p className="text-green-600 font-medium">Confidence</p>
                <p className="text-green-800">{multiModelResults.summary.confidence_level}</p>
              </div>
              <div>
                <p className="text-green-600 font-medium">Models Used</p>
                <p className="text-green-800">{Object.keys(multiModelResults.individual_results).length}</p>
              </div>
            </div>
          </div>

          {/* Fouling Categories */}
          <div className="mb-6">
            <h4 className="font-semibold text-gray-800 mb-3">Fouling Categories Detected</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(multiModelResults.summary.fouling_categories).map(([category, count]) => (
                <div key={category} className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="font-medium text-blue-800">{category}</p>
                  <p className="text-sm text-blue-600">{count} items</p>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendation */}
          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h4 className="font-semibold text-yellow-800 mb-2">AI Recommendation</h4>
            <p className="text-yellow-700">{multiModelResults.summary.recommendation}</p>
          </div>

          {/* Individual Model Results */}
          <div className="mt-6">
            <h4 className="font-semibold text-gray-800 mb-3">Individual Model Results</h4>
            <div className="space-y-4">
              {Object.entries(multiModelResults.individual_results).map(([modelName, result]) => (
                <div key={modelName} className="p-4 border border-gray-200 rounded-lg">
                  <h5 className="font-medium text-gray-800 mb-2">
                    {modelName} {result.success ? '✅' : '❌'}
                  </h5>
                  {result.success && result.classifications && (
                    <div className="text-sm text-gray-600">
                      <p>Marine items: {result.classifications.filter(c => c.is_marine_related).length}</p>
                      <p>Top classification: {result.classifications[0]?.label} ({result.classifications[0]?.confidence})</p>
                    </div>
                  )}
                  {result.error && (
                    <p className="text-sm text-red-600">Error: {result.error}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Brain className="w-6 h-6 mr-2 text-purple-600" />
          AI Image Classification
        </h2>
        <p className="text-gray-600 mb-6">
          Upload an image to classify marine biofouling using multiple AI models. 
          Get detailed analysis of fouling types, confidence scores, and recommendations.
        </p>

        {/* File Upload */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          {selectedFile ? (
            <div>
              <p className="text-green-600 font-medium">{selectedFile.name}</p>
              <p className="text-sm text-gray-500 mt-1">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div>
              <p className="text-lg font-medium text-gray-700">
                {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
              </p>
              <p className="text-gray-500 mt-2">or click to select a file</p>
              <p className="text-sm text-gray-400 mt-1">
                Supports: JPEG, PNG, GIF, BMP, WebP
              </p>
            </div>
          )}
        </div>

        {/* Model Selection */}
        {selectedFile && (
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model for Single Classification:
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {models.map((model) => (
                <option key={model.value} value={model.value}>
                  {model.label}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Action Buttons */}
        {selectedFile && (
          <div className="mt-6 flex space-x-4">
            <button
              onClick={classifySingleModel}
              disabled={isLoading}
              className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              ) : (
                <Brain className="w-5 h-5 mr-2" />
              )}
              Classify with {models.find(m => m.value === selectedModel)?.label}
            </button>

            <button
              onClick={classifyMultiModel}
              disabled={isLoading}
              className="flex-1 bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              ) : (
                <Brain className="w-5 h-5 mr-2" />
              )}
              Multi-Model Analysis
            </button>
          </div>
        )}
      </div>

      {/* Results */}
      {renderClassificationResults()}
    </div>
  );
};

export default ImageClassificationPanel;
