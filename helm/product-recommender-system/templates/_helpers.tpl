{{/*
Expand the name of the chart.
*/}}
{{- define "product-recommender-system.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "product-recommender-system.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "product-recommender-system.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "product-recommender-system.labels" -}}
helm.sh/chart: {{ include "product-recommender-system.chart" . }}
{{ include "product-recommender-system.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "product-recommender-system.selectorLabels" -}}
app.kubernetes.io/name: {{ include "product-recommender-system.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "product-recommender-system.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "product-recommender-system.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "product-recommender-system.feastEnv" -}}
- name: FEAST_PROJECT_NAME
  value: {{ .Values.feast.project }}
- name: FEAST_SECRET_NAME
  value: {{ .Values.feast.secret }}
- name: FEAST_REGISTRY_URL
  value: {{ .Values.feast.registry }}.{{ .Release.Namespace }}.svc.cluster.local
{{- end }}
