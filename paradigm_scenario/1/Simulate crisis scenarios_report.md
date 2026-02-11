# Use Case Description Report

## 1. Use Case Name
**Simulate crisis scenarios**

## 2. Description
Support crisis simulation for emergency services to test response plans.

## 3. Primary Actor
Emergency Services Officer

## 4. Problem Domain Context
The IT system must provide urban mobility planners with a dynamic simulation platform that:
1. Models traffic congestion under various scenarios.  
2. Allows real‑time adjustment of speed limits, reversible lane directions, roundabouts, adaptive traffic lights, road closures, and event‑induced traffic surges.  
3. Supplies predictive analytics on congestion, emissions, noise, and neighborhood impact.  
4. Triggers alerts when a measure creates congestion or inequity.  
5. Incorporates historical sensor data for scenario analysis.  
6. Estimates financial costs and environmental impacts.  
7. Supports crisis simulation for emergency services.  
8. Generates customizable reports (executive summaries, detailed analyses, fairness, cost, emission, and congestion metrics).  
9. Offers a recommendation engine for congestion‑mitigation solutions.

## 5. Preconditions
- The simulation engine is initialized and connected to real‑time traffic data feeds.  
- The Emergency Services Officer is authenticated and authorized to access the crisis simulation module.  
- Historical sensor data for the target area is available in the data lake.  
- The system’s alerting and recommendation components are operational.

## 6. Postconditions
- The crisis scenario is executed and the simulation results are stored in the system.  
- All relevant reports and analytics are generated and accessible to the actor.  
- Any alerts or recommendations triggered during the simulation are logged and notified to the appropriate stakeholders.

## 7. Main Flow
1. **Emergency Services Officer** selects the *Simulate crisis scenarios* function from the dashboard.  
2. The system presents a scenario template library; the officer chooses a predefined crisis scenario or creates a custom one.  
3. The officer configures scenario parameters (e.g., accident location, road closure extent, event surge intensity).  
4. The system validates the input against data integrity and policy constraints.  
5. The system loads the relevant historical sensor data for the selected area and time window.  
6. The simulation engine initializes the traffic model with the configured parameters and historical base state.  
7. The simulation runs iteratively, updating traffic flow, speed limits, lane directions, and adaptive lights in real time.  
8. During each iteration, the system monitors congestion, emissions, noise, and neighborhood impact metrics.  
9. If any metric exceeds predefined thresholds, the system triggers an alert and logs the incident.  
10. Concurrently, the recommendation engine evaluates mitigation actions and suggests options to the officer.  
11. Once the simulation completes, the system aggregates results and generates customizable reports (executive summary, detailed analysis, fairness, cost, emission, congestion).  
12. The officer reviews the reports and, if desired, exports them or shares them with stakeholders.  
13. The system archives the simulation run and updates the scenario library with any new insights.

## 8. Alternative Flows
- **AF1 (Custom Scenario Validation Failure)** – *Branch from step 5*:  
  1. The system detects invalid or incomplete parameters (e.g., missing road closure coordinates).  
  2. The system prompts the officer to correct the input and re‑validate.  
  3. The officer updates the parameters and the flow returns to step 5.

- **AF2 (Historical Data Unavailable)** – *Branch from step 5*:  
  1. The system cannot retrieve required sensor data for the selected area.  
  2. The system informs the officer and offers to use a fallback dataset or cancel the simulation.  
  3. If the officer chooses a fallback dataset, the flow proceeds to step 6.

- **AF3 (Simulation Engine Timeout)** – *Branch from step 7*:  
  1. The engine exceeds the maximum allowed runtime.  
  2. The system aborts the simulation, logs the timeout, and notifies the officer.  
  3. The officer may retry with adjusted parameters or cancel the run.

## 9. Exceptions
- **E1 (Authentication Failure)** – *Occurs at step 1*:  
  - The system rejects access, logs the event, and displays an error message prompting the user to re‑authenticate.

- **E2 (Data Feed Disruption)** – *Occurs during steps 5 or 7*:  
  - The system detects loss of real‑time traffic data.  
  - It switches to a degraded mode using cached data, logs the disruption, and informs the officer.

- **E3 (Alert Suppression Policy Violation)** – *Occurs during step 9*:  
  - An alert is suppressed due to conflicting policy rules.  
  - The system logs the suppression, notifies the officer, and provides the reason.

- **E4 (Report Generation Failure)** – *Occurs during step 11*:  
  - The report engine encounters an unexpected error.  
  - The system logs the error, sends a notification to the officer, and offers to retry or export raw data.

- **E5 (Recommendation Engine Crash)** – *Occurs during step 10*:  
  - The recommendation service becomes unavailable.  
  - The system logs the outage, notifies the officer, and continues the simulation without recommendations.

---