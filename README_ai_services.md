# AgentForge AI Services

AI Services is the AI-facing service layer of **AgentForge**. It provides the AI-service functionality used by the wider AgentForge platform and is designed to be consumed by the backend and other platform components.

> **Project status:** `agentforge-ai-services` was completed, functionality-tested, Dockerized, and its Docker deployment was confirmed working as of **17 August 2026**.

## Overview

The `agentforge-ai-services` repository is responsible for the AI-service layer within the AgentForge multi-repository architecture.

Its role is to provide a dedicated service boundary for AI-related capabilities so that the main AgentForge backend does not need to contain all AI-service implementation details directly.

The repository is part of the canonical AgentForge architecture:

```text
AgentForge
├── agentforge-backend
├── agentforge-frontend
├── agentforge-agents
├── agentforge-ai-services
├── agentforge-integrations
├── agentforge-docs
├── agentforge-infra
└── agentforge-shared
```

## Role in AgentForge

The high-level relationship between the major components is:

```text
                    AgentForge Frontend
                           |
                           v
                    AgentForge Backend
                     /            \
                    v              v
          AgentForge Agents    AI Services
                                   |
                                   v
                            AI/Model Layer
```

`agentforge-ai-services` is therefore kept as a separate repository/service so AI functionality can evolve independently from the core backend and agent definitions.

## Repository Status

The repository has already gone through its completion verification.

| Verification Area | Status |
|---|---|
| AI Services functionality | ✅ Tested |
| Dockerization | ✅ Completed |
| Docker deployment | ✅ Confirmed working |
| Repository status | ✅ COMPLETE |

The repository should be treated as a completed component unless a new requirement, regression, or integration issue is discovered.

## Docker

Docker support was completed and deployment was successfully verified.

Build the service using the repository's Docker configuration:

```powershell
docker build -t agentforge-ai-services .
```

If a Compose configuration is provided in the repository, use:

```powershell
docker compose up -d --build
```

Check running containers with:

```powershell
docker compose ps
```

View service logs with:

```powershell
docker compose logs -f
```

Stop the service with:

```powershell
docker compose down
```

> Use the repository's existing Docker/Compose configuration as the source of truth for the exact image name, ports, environment variables, and service names.

## Configuration

Configuration should be supplied through the environment mechanism defined by the repository.

Do not commit credentials, API keys, tokens, or other secrets to Git.

For local development, create the repository's expected environment file from its provided example/configuration template, if available.

Typical AI-service deployments may require model-provider configuration, but the exact variables should be taken from the repository's current configuration files rather than guessed.

## Running the Service

The recommended deployment path is Docker because it provides the same service boundary used during the completed deployment verification.

Typical workflow:

```powershell
docker compose up -d --build
```

Then verify:

```powershell
docker compose ps
```

and inspect logs if required:

```powershell
docker compose logs -f
```

If the repository provides a documented native Python entry point, that entry point should be preferred for development-only execution.

## Testing

The AI Services repository was functionally tested before being marked complete.

The verification goal was to confirm that:

1. The service can start successfully.
2. AI-service functionality works as expected.
3. The service can be packaged into Docker.
4. The Dockerized service can be deployed successfully.

The repository was subsequently marked **COMPLETE** after those checks succeeded.

## Integration with AgentForge

The AI Services repository is not intended to replace the AgentForge agent layer.

The responsibilities are separated conceptually:

```text
agentforge-agents
        |
        | Agent behavior / agent configuration
        v
agentforge-backend
        |
        | API / orchestration
        v
agentforge-ai-services
        |
        | AI-service functionality
        v
      Models
```

This separation helps keep:

- agent behavior in `agentforge-agents`
- API/business orchestration in `agentforge-backend`
- AI-service implementation in `agentforge-ai-services`
- shared primitives in `agentforge-shared`

## Deployment Verification

Docker deployment was explicitly verified successfully.

The completion milestone therefore includes:

```text
Source
  ↓
Build
  ↓
Docker image
  ↓
Container
  ↓
Service starts
  ↓
Functionality verified
  ↓
Deployment confirmed
```

## Development Guidelines

### Keep secrets out of source control

Never commit:

```text
API keys
access tokens
passwords
private credentials
production secrets
```

Use environment variables or the project's configured secret-management mechanism.

### Preserve service boundaries

AI-specific implementation should remain inside this repository rather than being duplicated inside the backend.

When adding new functionality, consider whether it belongs in:

- AI Services
- Agents
- Backend
- Integrations
- Shared

before implementing it.

### Test before changing completed functionality

This repository has already been verified and Dockerized. Avoid unnecessary refactoring of working AI-service code unless there is a clear requirement or regression.

## Troubleshooting

### Container does not start

Check:

```powershell
docker compose ps
docker compose logs
```

Then verify that all required environment variables are configured.

### AI provider/API errors

Check the service environment configuration and credentials. Do not place credentials directly into source files.

### Docker build problems

Rebuild without using stale layers when necessary:

```powershell
docker compose build --no-cache
docker compose up -d
```

### Service connectivity problems

Check:

```powershell
docker compose ps
docker compose logs -f
```

and verify that the service is using the expected Docker network and configuration.

## AgentForge Repository Completion Status

As of 19 August 2026:

```text
agentforge-shared          ✅ COMPLETE
agentforge-ai-services    ✅ COMPLETE
agentforge-agents         ✅ COMPLETE
agentforge-integrations   ✅ COMPLETE
agentforge-backend        ✅ COMPLETE
agentforge-frontend       ⏳ PENDING
agentforge-docs           ⏳ PENDING
agentforge-infra          ⏳ PENDING
```

## Completion Record

`agentforge-ai-services` was completed by **Ajay**.

The completed milestone includes:

- AI Services functionality testing
- Dockerization
- Successful Docker deployment verification

The service is now a verified baseline component of AgentForge.

## Next Work

Do not reopen this repository for routine changes unless:

- a regression is discovered,
- an integration requirement changes,
- a new AI capability needs to be added, or
- deployment/configuration requirements change.

The current AgentForge project focus should move to the remaining repositories, with `agentforge-docs` being the next recommended focus.

---

**AgentForge AI Services — Completed and verified by Ajay**  
**Status: COMPLETE**
