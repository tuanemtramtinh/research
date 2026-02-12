```markdown
# Use Case Description Report

## 1. Use Case Name
**Estimate financial and environmental impacts**

## 2. Description
Calculate costs and environmental impacts of proposed traffic measures.

## 3. Primary Actor
Urban Mobility Planner

## 4. Problem Domain Context
The IT system must provide urban mobility planners with a dynamic simulation platform that:

1. **Models traffic congestion** under various scenarios.  
2. **Allows real‑time adjustment** of speed limits, reversible lane directions, roundabouts, adaptive traffic lights, road closures, and event‑induced traffic surges.  
3. **Supplies predictive analytics** on congestion, emissions, noise, and neighborhood impact.  
4. **Triggers alerts** when a measure creates congestion or inequity.  
5. **Incorporates historical sensor data** for scenario analysis.  
6. **Estimates financial costs** and environmental impacts.  
7. **Supports crisis simulation** for emergency services.  
8. **Generates customizable reports** (executive summaries, detailed analyses, fairness, cost, emission, and congestion metrics).  
9. **Offers a recommendation engine** for congestion‑mitigation solutions.

## 5. Preconditions
- None.

## 6. Postconditions
- None.

## 7. Main Flow
1. **Urban Mobility Planner** selects the “Estimate financial and environmental impacts” use case from the dashboard.  
2. System prompts the planner to **choose a traffic measure** (e.g., new speed limit, reversible lane, roundabout).  
3. Planner **inputs measure parameters** (location, duration, target speed, lane configuration, etc.).  
4. System **validates input data** against data integrity rules.  
5. System **retrieves relevant historical sensor data** for the selected area and time frame.  
6. System **runs the simulation engine** with the specified measure and current traffic state.  
7. Simulation produces **short‑term traffic projections** (congestion, travel time, queue lengths).  
8. System **calculates financial cost** using predefined cost models (construction, maintenance, operational).  
9. System **computes environmental impacts** (emissions, noise, air quality) using emission factors and traffic volumes.  
10. System **aggregates results** into a structured report format.  
11. Planner **reviews the preliminary report** on the screen.  
12. Planner chooses to **customize the report** (select metrics, add executive summary, adjust layout).  
13. System **generates the final report** and offers options to **export** (PDF, Excel, share link).  
14. Planner **saves or archives** the report in the project repository.  
15. System **logs the activity** for audit purposes and updates the planner’s dashboard with the new estimate.

## 8. Alternative Flows
- **AF1 (Step 5 – Data Retrieval Failure):**  
  1. System fails to retrieve historical data.  
  2. System alerts the planner with an error message.  
  3. Planner can **retry** or **continue with a generic dataset**.  

- **AF2 (Step 9 – Environmental Model Unavailable):**  
  1. The required emission factor model is missing.  
  2. System prompts the planner to **select an alternative model** or **input custom factors**.  

- **AF3 (Step 12 – Customization Cancelled):**  
  1. Planner cancels report customization.  
  2. System reverts to the default report template and skips to Step 13.

## 9. Exceptions
- **E1 (Step 4 – Validation Error):**  
  - Input values fall outside acceptable ranges.  
  - System displays specific error messages and prevents progression until corrected.  

- **E2 (Step 6 – Simulation Timeout):**  
  - Simulation exceeds maximum allowed runtime.  
  - System aborts the run, logs the timeout, and offers to **reduce scenario complexity** or **increase resource allocation**.  

- **E3 (Step 8 – Cost Model Incomplete):**  
  - Missing cost parameters for the selected measure.  
  - System notifies the planner and provides a **fallback estimation** using average cost per kilometer or a user‑defined placeholder.  

- **E4 (Step 10 – Report Generation Failure):**  
  - System encounters an internal error while compiling the report.  
  - System rolls back to the previous state, saves a **partial report** (if available), and logs the error for support.  

- **E5 (Step 13 – Export Failure):**  
  - Export format is unsupported or destination is unreachable.  
  - System offers alternative formats or retries with a different destination.  

```
