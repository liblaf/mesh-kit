# SCULPTOR

## Pre-Processing

```mermaid
flowchart TD
  subgraph Template
    S0[00]
    S1[01]
    S2[02]
    S3[03]
    S3S[03 + sparse landmarks]
    S3 -->|manual selection| S3S
    S0 -->|preprocessing| S1 -->|manual refine| S2 -->|MeshFix| S3
  end
  subgraph Target
    T0[00-CT]
    T1[01]
    T1S[01 + sparse landmarks]
    T1D[01 + dense landmarks]
    T2S[02 + sparse landmarks]
    T2D[02 + dense landmarks]
    T4[04]
    T0 -->|CT to mesh| T1 -->|manual selection| T1S
    S3S & T1S -->|align| T2S
    T2S & T1S -->|densify| T2D & T1D
    T2D & T1D -->|register| T4
  end
```
