# Use Case Description Report

## 1. Use Case Name
**Ingest Sensor Data**

## 2. Description
The system receives live sensor feeds and updates the simulation in real time.

## 3. Primary Actor
**System Admin**

## 4. Problem Domain Context
The IT system must provide a real‑time, map‑based traffic simulator for Wonderland that ingests live sensor data, displays current traffic flows, heat maps, and environmental metrics (CO₂, noise, air pollution) on the map; it must store all current and historical simulation states (map, traffic, heat maps, environmental readings) tied to timestamps, allow users to query, compare, and analyze traffic and environmental data across arbitrary time periods, including side‑by‑side comparisons of two moments or simulation predictions versus real measurements; it must support adding new data sources and sensor feeds on demand, enable predictive modeling of future traffic and environmental outcomes based on historical patterns, and allow users to capture snapshots of specific times for simulation runs—all presented through an intuitive, self‑service interface that does not require expert assistance.

## 5. Preconditions
- The system is online and the simulation engine is running.
- Sensor data sources are registered in the system configuration.
- The database contains the schema for storing sensor readings and simulation states.
- Network connectivity to sensor feeds is established.

## 6. Postconditions
- Sensor data is stored with accurate timestamps.
- The simulation map and associated visualizations are updated to reflect the new data.
- Alerts or notifications are generated for any anomalous readings.
- Historical data archive is extended to include the newly ingested data.

## 7. Main Flow
1. **System Admin** initiates the ingestion process by enabling a new sensor feed through the admin console.
2. The system authenticates the sensor feed and verifies its data format against the schema registry.
3. Upon successful authentication, the system establishes a persistent connection to the sensor’s streaming endpoint.
4. The sensor begins streaming data packets (e.g., GPS coordinates, speed, CO₂ levels, noise levels) at defined intervals.
5. The system receives each packet, validates integrity, and parses the payload into structured records.
6. Parsed records are timestamped and written to the real‑time ingestion pipeline (e.g., Kafka, Pulsar).
7. The ingestion pipeline forwards records to the simulation engine in batches or as a stream, depending on configuration.
8. The simulation engine processes incoming data, updating traffic flow models, heat maps, and environmental metrics in real time.
9. Updated simulation states are persisted to the database with corresponding timestamps.
10. The map‑based UI refreshes, displaying the most recent traffic flows, heat maps, and environmental overlays.
11. If any data points exceed predefined thresholds, the system triggers alerts and logs them in the incident management module.
12. The System Admin observes the live updates and can pause or stop the feed if necessary.

## 8. Alternative Flows
1. **Sensor Feed Unavailable**  
   1.1 The system fails to establish a connection to the sensor endpoint.  
   1.2 The system logs the failure, retries after a back‑off period, and alerts the admin.  
2. **Data Format Mismatch**  
   3.1 The system detects that the incoming packet does not conform to the expected schema.  
   3.2 The packet is discarded, an error is logged, and a notification is sent to the admin for investigation.  
3. **High Latency or Packet Loss**  
   6.1 The ingestion pipeline detects a drop in packet arrival rate.  
   6.2 The system switches to a buffered mode, temporarily storing data locally until connectivity is restored.  
   6.3 Once normal flow resumes, buffered data is replayed to the simulation engine.  

## 9. Exceptions
1. **Authentication Failure** – If the sensor feed fails authentication (step 2), the system logs the error, prevents further connection attempts, and notifies the admin.  
2. **Data Corruption** – If a packet fails integrity checks (step 5), the system discards the packet, increments a corruption counter, and alerts the admin after a threshold is reached.  
3. **Pipeline Back‑pressure** – If the ingestion pipeline reaches capacity (step 6), the system temporarily throttles incoming data, signals the sensor to slow its transmission rate, and queues excess data.  
4. **Simulation Engine Crash** – If the simulation engine crashes while processing data (step 8), the system redirects the pipeline to a standby instance, logs the incident, and restores normal processing.  
5. **Database Write Failure** – If persisting simulation state fails (step 9), the system retries up to three times, then rolls back the current batch, logs the failure, and alerts the admin.  

---