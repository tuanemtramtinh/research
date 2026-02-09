**Use Case Name**  
Share Wearable Data  

**Description**  
Enables a patient to upload wearable‑device data and selectively grant or revoke access to specific healthcare providers.  

**Primary Actor**  
Patient  

**Problem Domain Context**  
Develop an integrated patient‑care platform that:  

1. Provides patients with secure mobile access to their full medical record, including measurement, test results and prescription renewal.  
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers.  
3. Allows doctors to view, edit and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility.  
4. Permits nurses to edit and review records for clinical preparation.  
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability.  
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes.  
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details.  
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.  

**Preconditions**  
- Patient has a registered account on the platform.  
- Patient has granted the application permission to access wearable device data via the device’s API or SDK.  
- Wearable device is paired and actively transmitting data.  

**Postconditions**  
- The selected wearable data is securely stored in the patient’s health record.  
- Specified healthcare providers have updated access permissions (granted or revoked).  
- Audit log records the sharing action with timestamps and provider identities.  

**Main Flow**  
1. **Patient** logs into the mobile application using two‑step authentication.  
2. **Patient** navigates to the “Wearable Data” section of their health record.  
3. **System** displays available wearable devices and the most recent data streams.  
4. **Patient** selects a device and chooses to upload the latest data set.  
5. **System** validates data integrity and encrypts the payload before storage.  
6. **System** presents a list of healthcare providers currently linked to the patient’s account.  
7. **Patient** selects one or more providers and sets the desired access level (View / Comment / Full Edit).  
8. **System** updates the access control list and sends push notifications to the selected providers.  
9. **System** records the action in the audit log.  
10. **Patient** receives a confirmation message indicating successful data upload and sharing.  

**Alternative Flows**  
- **4a.** *No data available from device*  
  1. **System** displays an error message: “No new data from device.”  
  2. **Patient** can retry or cancel.  

- **7a.** *Patient chooses to revoke access*  
  1. **Patient** selects a provider and toggles the access level to “Revoked.”  
  2. **System** removes the provider’s permissions and sends a revocation notification.  

**Exceptions**  
- **5.** *Data validation fails*  
  1. **System** aborts the upload, displays “Data integrity check failed.”  
  2. **Patient** is prompted to retry or contact support.  

- **6.** *Device pairing error*  
  1. **System** shows “Unable to retrieve device list. Please reconnect.”  
  2. **Patient** attempts to re‑pair the device.  

- **7.** *Provider not found in system*  
  1. **System** alerts “Selected provider does not exist in your network.”  
  2. **Patient** can search again or add a new provider.  

- **10.** *Push notification failure*  
  1. **System** logs the failure and retries up to three times.  
  2. **Patient** receives an email confirmation if notifications fail.  

---