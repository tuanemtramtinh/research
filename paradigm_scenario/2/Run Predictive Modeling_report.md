# Run Predictive Modeling

**Use Case Name:** Run Predictive Modeling  
**Description:** The system uses historical data to forecast future traffic and environmental outcomes, presenting predictions alongside real measurements.  
**Primary Actor:** Traffic Analyst  
**Problem Domain Context:**  
The IT system must provide a real‑time, map‑based traffic simulator for Wonderland that ingests live sensor data, displays current traffic flows, heat maps, and environmental metrics (CO₂, noise, air pollution) on the map; it must store all current and historical simulation states (map, traffic, heat maps, environmental readings) tied to timestamps, allow users to query, compare, and analyze traffic and environmental data across arbitrary time periods, including side‑by‑side comparisons of two moments or simulation predictions versus real measurements; it must support adding new data sources and sensor feeds on demand, enable predictive modeling of future traffic and environmental outcomes based on historical patterns, and allow users to capture snapshots of specific times for simulation runs—all presented through an intuitive, self‑service interface that does not require expert assistance.

## Preconditions
- Historical sensor data for the selected time range is available and properly ingested into the database.  
- The Traffic Analyst has authenticated and has sufficient permissions to run predictive models.  
- The system’s predictive engine is configured with at least one valid model (e.g., ARIMA, LSTM).  

## Postconditions
- A new simulation snapshot is created, containing predicted traffic flow and environmental metrics for the specified future period.  
- The results are stored in the database and are accessible for future queries and visualizations.  
- The user receives a confirmation notification (e.g., email or UI message) indicating successful completion.

## Main Flow
1. **Traffic Analyst** selects the *Run Predictive Modeling* option from the dashboard.  
2. The system prompts the Analyst to specify:  
   - Time range for historical data to use.  
   - Target forecast horizon (e.g., next 24 h).  
   - Desired output metrics (traffic volume, CO₂, noise, etc.).  
3. The Analyst inputs the parameters and submits the request.  
4. The system validates the input values and checks model availability.  
5. The system retrieves the required historical data from the time‑series store.  
6. The predictive engine processes the data using the selected model(s).  
7. The system generates forecasted values for each metric at each forecast time step.  
8. The system creates a new simulation snapshot, tagging it with:  
   - Timestamp of creation.  
   - Forecast horizon.  
   - Model metadata (name, version, parameters).  
9. The snapshot is persisted in the database and indexed for retrieval.  
10. The system updates the user interface to display:  
    - A comparison panel showing real measurements versus predictions.  
    - Heat maps for predicted traffic and environmental conditions.  
11. The **Traffic Analyst** reviews the results and may export them (CSV, PDF) or save the snapshot for later use.  
12. System logs the action and sends a completion notification to the Analyst.

## Alternative Flows
1. **Model Selection Failure**  
   - *Trigger:* Step 4 fails because no suitable model is configured.  
   - *Action:* The system presents a list of available models for the Analyst to choose from or offers to create a new model.  
2. **Data Retrieval Timeout**  
   - *Trigger:* Step 5 exceeds the maximum allowed time.  
   - *Action:* The system aborts the request, logs the timeout, and notifies the Analyst to retry later.  
3. **User Cancels Operation**  
   - *Trigger:* Step 3 (submission) is canceled by the Analyst.  
   - *Action:* The system discards any in‑progress data and returns to the dashboard without creating a snapshot.

## Exceptions
1. **Insufficient Permissions** – *Step 1*  
   - The system denies access and shows an error message: “You do not have permission to run predictive models.”  
2. **Invalid Input Values** – *Step 3*  
   - The system highlights erroneous fields and prompts the Analyst to correct them before submission.  
3. **Model Execution Error** – *Step 7*  
   - If the predictive engine throws an exception, the system logs the error, rolls back any partial snapshot creation, and displays: “Forecast generation failed. Please check model configuration and retry.”  
4. **Database Write Failure** – *Step 9*  
   - The system attempts a retry; upon repeated failure, it alerts the Analyst and rolls back the operation, ensuring no corrupted data is stored.  
5. **Network Disruption** – *Any step involving external services*  
   - The system queues the request for retry when connectivity is restored and informs the Analyst of the temporary outage.