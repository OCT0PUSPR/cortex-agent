{{/*
Expand the name of the chart.
*/}}
{{- define "cortex-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this
(by the DNS naming spec).
*/}}
{{- define "cortex-agent.fullname" -}}
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
{{- define "cortex-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cortex-agent.labels" -}}
helm.sh/chart: {{ include "cortex-agent.chart" . }}
{{ include "cortex-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: cortex-agent
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cortex-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cortex-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
The image reference (repository:tag). Tag defaults to the chart appVersion.
*/}}
{{- define "cortex-agent.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}

{{/*
The name of the Secret to use (existing one, or the chart-managed one).
*/}}
{{- define "cortex-agent.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secret" (include "cortex-agent.fullname" .) }}
{{- end }}
{{- end }}

{{/*
The name of the ConfigMap.
*/}}
{{- define "cortex-agent.configMapName" -}}
{{- printf "%s-config" (include "cortex-agent.fullname" .) }}
{{- end }}
