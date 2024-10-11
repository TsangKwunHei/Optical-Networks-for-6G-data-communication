---
config:
  theme: dark
  look: classic
  layout: elk
---
graph TD
    subgraph External Inputs
        A[Optical Network Data] --> B[InitializeEnvironment]
        A1[Service Details] --> B[InitializeEnvironment]
        A2[Edge Details] --> B[InitializeEnvironment]
    end
    subgraph Environment Initialization
        B[InitializeEnvironment] --> B1[Read Nodes and Edges]
        B1 --> B2[Set Conversion Opportunities]
        B2 --> B3[Create Adjacency List]
        B3 --> B4[Read Initial Services]
        B4 --> B5[Initialize Edge Wavelengths]
    end
    subgraph Service Management
        B --> C[Service Management]
        C --> C1[Detect Affected Services]
        C1 --> C1A[Check Service Paths]
        C1A --> C1B[Mark Services for Replanning]
        C --> C2[Handle Edge Failure]
        C2 --> C2A[Remove Edge from Adjacency List]
        C2A --> C2B[Update Edge List and Adjacency]
        C --> C3[Assign Wavelengths Using GA]
        C3 --> C3A[Create Temp Edge Wavelengths]
        C3A --> C3B[Create Temp Pi Values]
        C3B --> C3C[Run GA for Services]
    end
    subgraph Genetic Algorithm
        C3C --> D[Run Genetic Algorithm]
        D --> D1[Generate Initial Population]
        D1 --> D1A[Evaluate Initial Chromosomes]
        D1A --> D1B[Store Fitness Values]
        D --> D2[Selection]
        D2 --> D2A[Tournament Selection]
        D --> D3[Crossover & Mutation]
        D3 --> D3A[Crossover Chromosomes]
        D3A --> D3B[Mutate Chromosomes]
        D --> D4[Evaluate New Population]
        D4 --> D4A[Compute Fitness for New Generation]
    end
    subgraph Replan Services
        C3C --> E[Replan Services]
        E --> E1[Check Available Wavelengths]
        E1 --> E2[Update Service Paths]
        E2 --> E3[Update Edge Wavelengths Globally]
    end
    subgraph Min Cut Computation
        F[Compute Min Cut] --> F1[Run Dinic's Algorithm]
        F1 --> F2[Calculate Max Flow]
        F2 --> F3[Identify Min Cut Edges]
        F3 --> F4[Generate Failure Scenarios]
    end
    subgraph Data Flow
        B -->|Edge Data| F
        F -->|Failure Scenarios| D
        D -->|Replanning Data| E
        E -->|Service Update| C
        C -->|Results| G[Save Results]
    end
    G -->|Final Output| ExternalEntities[Saved Configurations]
