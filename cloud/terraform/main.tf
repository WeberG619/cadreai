terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Artifact Registry repository for container images
resource "google_artifact_registry_repository" "cadre" {
  location      = var.region
  repository_id = "cadre-ai"
  format        = "DOCKER"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "cadre" {
  name     = "cadre-ai"
  location = var.region

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/cadre-ai/cadre-ai:${var.image_tag}"

      ports {
        container_port = 8080
      }

      env {
        name  = "GOOGLE_API_KEY"
        value = var.google_api_key
      }
      env {
        name  = "REVIT_ENABLED"
        value = "false"
      }
      env {
        name  = "CADRE_MODEL"
        value = var.cadre_model
      }
      env {
        name  = "PORT"
        value = "8080"
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "1Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/status"
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    # 30-minute timeout for long voice sessions
    timeout = "1800s"

    # Session affinity keeps WebSocket connections on the same instance
    session_affinity = true
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Allow unauthenticated access
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = google_cloud_run_v2_service.cadre.project
  location = google_cloud_run_v2_service.cadre.location
  name     = google_cloud_run_v2_service.cadre.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "service_url" {
  value       = google_cloud_run_v2_service.cadre.uri
  description = "Cloud Run service URL"
}
