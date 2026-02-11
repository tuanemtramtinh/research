## Use Case Name
Display Traffic and Environmental Heat Maps

## Description
The system renders current traffic flows, heat maps, and environmental metrics on an interactive map for users.

## Primary Actor
Regular User

## Problem Domain Context
The IT system must provide a real‑time, map‑based traffic simulator for Wonderland that ingests live sensor data, displays current traffic flows, heat maps, and environmental metrics (CO₂, noise, air pollution) on the map; it must store all current and historical simulation states (map, traffic, heat maps, environmental readings) tied to timestamps, allow users to query, compare, and analyze traffic and environmental data across arbitrary time periods, including side‑by‑side comparisons of two moments or simulation predictions versus real measurements; it must support adding new data sources and sensor feeds on demand, enable predictive modeling of future traffic and environmental outcomes based on historical patterns, and allow users to capture snapshots of specific times for simulation runs—all presented through an intuitive, self‑service interface that does not require expert assistance.

## Preconditions
None.

## Postconditions
The system displays the requested heat maps and traffic data on the interactive map; all user actions and displayed data states are logged and stored with corresponding timestamps.

## Main Flow
1. **Actor initiates action.** The Regular User opens the dashboard and selects the “Display Traffic and Environmental Heat Maps” feature.  
2. **System retrieves current sensor data.** The backend queries live traffic sensors, CO₂, noise, and air pollution sensors for the selected area.  
3. **System aggregates data.** Real‑time data streams are aggregated and harmonized into a unified spatial-temporal representation.  
4. **System generates heat maps.** The aggregated data is rendered into heat map layers for traffic density, CO₂ concentration, noise levels, and air pollution.  
5. **System overlays traffic flows.** Current traffic flow vectors (speed, direction) are plotted over the heat maps.  
6. **System displays interactive map.** The user sees an interactive map with zoom, pan, and layer toggle capabilities.  
7. **Actor interacts with map.** The user can click on specific zones to view detailed metrics, adjust time sliders, or add comparison layers.  
8. **System updates view in real time.** As the user manipulates controls, the system refreshes the heat maps and traffic flows accordingly.  
9. **Actor saves snapshot (optional).** The user can capture the current view as a snapshot for later analysis or sharing.  
10. **System logs the snapshot.** The snapshot metadata (timestamp, layers, user ID) is stored for future retrieval.  
11. **Actor ends session.** The user closes the dashboard or logs out.

## Alternative Flows
1. **Alternative 1 – Historical Data Query.**  
   - *Trigger:* The user selects a past timestamp instead of “Current.”  
   - *Steps:*  
     1. System retrieves stored historical simulation state for the selected time.  
     2. Steps 5–9 proceed with the historical data instead of live data.  

2. **Alternative 2 – Side‑by‑Side Comparison.**  
   - *Trigger:* The user opts for a dual‑view comparison.  
   - *Steps:*  
     1. System opens two map panes side by side.  
     2. Each pane follows the main flow for its respective timestamp or simulation prediction.  
     3. User can toggle synchronization of zoom/pan between panes.

## Exceptions
1. **Exception 1 – Sensor Failure.**  
   - *Occurs at:* Step 2.  
   - *Handling:* The system displays a warning indicator for missing data sources and continues rendering available data with a “partial data” message.  

2. **Exception 2 – Data Aggregation Timeout.**  
   - *Occurs at:* Step 3.  
   - *Handling:* The system retries aggregation twice; if still failing, it reverts to the last successful aggregation snapshot and informs the user of a delay.  

3. **Exception 3 – Rendering Error.**  
   - *Occurs at:* Step 4–5.  
   - *Handling:* The system logs the error, displays a fallback static map, and offers to retry rendering after a brief pause.  

4. **Exception 4 – User Session Expired.**  
   - *Occurs at:* Any step after user interaction.  
   - *Handling:* The system prompts the user to log in again and restores the previous state upon successful authentication.