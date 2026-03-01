variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run deployment"
  type        = string
  default     = "us-central1"
}

variable "google_api_key" {
  description = "Google API key for Gemini"
  type        = string
  sensitive   = true
}

variable "cadre_model" {
  description = "Gemini model to use"
  type        = string
  default     = "gemini-2.5-flash-native-audio-latest"
}

variable "image_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}
