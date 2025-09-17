# Healthcare Appointment System with PydanticAI

This module demonstrates building a production-grade healthcare appointment system using PydanticAI, showcasing progressive complexity from basic agents to advanced self-correcting systems.

## Overview

The `healthcare_appointments.py` file contains 5 progressive examples that demonstrate key PydanticAI concepts applied to healthcare appointment management:

1. **Basic Agent** - Simple appointment inquiries
2. **Structured Responses** - Type-safe healthcare responses
3. **Dependencies & Context** - Patient information integration
4. **Tools Integration** - Appointment lookup and scheduling tools
5. **Self-Correction** - Validation and error handling

## Key Features

### Healthcare-Specific Models
- **PatientDetails**: Patient information with medical record numbers, insurance
- **Appointment**: Appointment scheduling with doctors and departments
- **HealthcareResponseModel**: Structured responses with urgency levels and department referrals

### HIPAA Compliance Features
- Patient identity validation
- Confidentiality checks
- Secure appointment access controls
- Professional response tone

### Medical Workflow Support
- **Urgency Classification**: routine, urgent, emergency
- **Department Routing**: Cardiology, Dermatology, etc.
- **Doctor Availability**: Scheduling coordination
- **Appointment Validation**: Self-correction for format errors

## How It Works

### 1. Basic Healthcare Agent
```python
agent1 = Agent(
    model=model,
    system_prompt="You are a helpful healthcare appointment assistant..."
)
```
Handles basic appointment inquiries with HIPAA-compliant responses.

### 2. Structured Healthcare Responses
```python
class HealthcareResponseModel(BaseModel):
    response: str
    urgency_level: str  # routine, urgent, emergency
    appointment_needed: bool
    department_referral: Optional[str]
```
Provides type-safe, structured responses for better clinical decision-making.

### 3. Patient Context Integration
```python
agent3 = Agent(
    deps_type=PatientDetails,
    output_type=HealthcareResponseModel
)
```
Injects patient information (MRN, insurance, appointment history) into agent context.

### 4. Healthcare Tools
- `get_appointment_details()`: Retrieves patient appointment information
- `check_doctor_availability()`: Checks doctor schedules
- `get_appointment_status()`: Validates appointment IDs
- `validate_patient_appointment()`: Ensures HIPAA compliance

### 5. Self-Correction & Validation
```python
@agent5.tool_plain()
def get_appointment_status(appointment_id: str) -> str:
    if appointment_info is None:
        raise ModelRetry("Please ensure appointment ID format (APT-XXXXX)")
```
Automatically corrects appointment ID formats and validates patient access rights.

## Running the Examples

```bash
cd src/
python healthcare_appointments.py
```

Each section will demonstrate:
- Basic appointment scheduling requests
- Emergency vs routine classification
- Patient context-aware responses
- Tool-assisted appointment management
- Self-correcting appointment validation

## Medical Use Cases Demonstrated

- **Appointment Scheduling**: "I need to see Dr. Smith next week"
- **Emergency Triage**: "I have severe chest pain" → urgency_level: "emergency"
- **Appointment Status**: "When is my next appointment?" → Patient context lookup
- **Doctor Availability**: "When is Dr. Smith available?" → Schedule checking
- **Appointment Validation**: "Status of appointment 12345" → Auto-corrects to "APT-12345"

## Key Benefits

1. **Type Safety**: Pydantic models ensure data validation
2. **HIPAA Compliance**: Built-in patient privacy protections
3. **Medical Accuracy**: Structured responses with urgency classification
4. **Self-Correction**: Automatic error handling and format validation
5. **Scalability**: Progressive complexity from basic to advanced agents

This example serves as a foundation for building real-world healthcare appointment systems with LLM integration while maintaining medical accuracy and patient confidentiality.