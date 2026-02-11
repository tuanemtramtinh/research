# Use Case Description Report

## 1. Use Case Name
**Generate customizable reports**

## 2. Description
Create executive summaries, detailed analyses, fairness, cost, emission, and congestion metrics reports.

## 3. Primary Actor
Urban Mobility Planner

## 4. Problem Domain Context
The IT system must provide urban mobility planners with a dynamic simulation platform that  
1. models traffic congestion under various scenarios,  
2. allows real‑time adjustment of speed limits, reversible lane directions, roundabouts, adaptive traffic lights, road closures, and event‑induced traffic surges,  
3. supplies predictive analytics on congestion, emissions, noise, and neighborhood impact,  
4. triggers alerts when a measure creates congestion or inequity,  
5. incorporates historical sensor data for scenario analysis,  
6. estimates financial costs and environmental impacts,  
7. supports crisis simulation for emergency services,  
8. generates customizable reports (executive summaries, detailed analyses, fairness, cost, emission, and congestion metrics), and  
9. offers a recommendation engine for congestion‑mitigation solutions.

## 5. Preconditions
None.

## 6. Postconditions
The system presents the requested report to the Urban Mobility Planner and stores a copy in the report repository for future reference.

## 7. Main Flow
1. **Actor initiates action** – The Urban Mobility Planner selects the *Generate customizable reports* option from the dashboard.  
2. **System prompts for report type** – The system displays a modal with available report categories (executive summary, detailed analysis, fairness, cost, emission, congestion metrics).  
3. **Actor selects desired categories** – The planner checks the boxes for the categories they wish to include.  
4. **System prompts for filters** – The system asks for time range, geographic area, scenario ID, and any custom parameters (e.g., specific road segments).  
5. **Actor inputs filters** – The planner enters or selects the required filter values.  
6. **System validates inputs** – The system checks that all required fields are populated and that the filters are consistent.  
7. **System aggregates data** – The system queries historical sensor data, simulation results, and cost/impact models to compile the requested metrics.  
8. **System generates report sections** – For each selected category, the system creates the corresponding report section using predefined templates.  
9. **System compiles full report** – All sections are assembled into a single document (PDF/HTML).  
10. **System displays preview** – The planner can preview the report in a viewer.  
11. **Actor reviews and customizes** – The planner can edit titles, add executive commentary, or adjust visualizations.  
12. **Actor finalizes report** – The planner clicks *Generate* to produce the final version.  
13. **System saves report** – The report is stored in the repository with metadata (author, creation time, filters used).  
14. **System offers export/share options** – The planner can email, download, or share the report via collaborative platform.  
15. **Use case completes** – The system returns to the dashboard.

## 8. Alternative Flows
- **AF1 – Missing filter values**  
  *Trigger:* Step 4 or 5  
  The system detects incomplete filters, displays an error message, and prompts the actor to complete the missing fields before proceeding.  
- **AF2 – No data available for selected criteria**  
  *Trigger:* Step 7  
  The system informs the actor that no records match the filters and offers to broaden the criteria or cancel the report generation.  
- **AF3 – Actor cancels before finalization**  
  *Trigger:* Step 11 or 12  
  The planner clicks *Cancel*; the system discards any unsaved changes and returns to the dashboard.  

## 9. Exceptions
1. **E1 – Data source failure during aggregation (Step 7)**  
   The system logs the error, displays a notification to the actor indicating that the report could not be generated due to a backend issue, and offers to retry.  
2. **E2 – Report generation timeout (Step 9)**  
   If the compilation exceeds the allowed time, the system aborts the process, informs the actor, and suggests reducing the scope or filters.  
3. **E3 – Permission denied when saving report (Step 13)**  
   The system alerts the actor that they lack write access to the repository, prompts for an alternative directory, or requests elevated permissions.  
4. **E4 – Export format unsupported (Step 14)**  
   If the actor selects an unsupported export format, the system notifies them and provides a list of supported formats.