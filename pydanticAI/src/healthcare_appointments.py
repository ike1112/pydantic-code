"""
Healthcare Appointment System using PydanticAI.

This module demonstrates how PydanticAI makes it easier to build
production-grade LLM-powered healthcare appointment systems with type safety and structured responses.
"""

from typing import Dict, List, Optional
from datetime import date
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.anthropic import AnthropicModel

from utils.markdown import to_markdown
from dotenv import load_dotenv

load_dotenv()

model = AnthropicModel("claude-3-7-sonnet-20250219")

# --------------------------------------------------------------
# 1. Simple Agent - Basic Healthcare Assistant
# --------------------------------------------------------------
"""
This example demonstrates the basic usage of PydanticAI agents for healthcare.
Key concepts:
- Creating a basic agent with a system prompt
- Running synchronous queries
- Accessing response data, message history, and costs
"""

agent1 = Agent(
    model=model,
    system_prompt="You are a helpful healthcare appointment assistant. Be professional, empathetic, and follow HIPAA guidelines.",
)

# Example: Basic appointment inquiry
if __name__ == "__main__":
    response = agent1.run_sync("I need to schedule an appointment with Dr. Smith for next week.")
    print(f"Response 1: {response.output}")

    response2 = agent1.run_sync(
        user_prompt="What was my previous request?",
        message_history=response.new_messages(),
    )
    print(f"Response 2: {response2.output}")

# --------------------------------------------------------------
# 2. Agent with Structured Response
# --------------------------------------------------------------
"""
This example shows how to get structured, type-safe responses from the healthcare agent.
Key concepts:
- Using Pydantic models to define response structure
- Type validation and safety
- Field descriptions for better model understanding
"""


class HealthcareResponseModel(BaseModel):
    """Structured response for healthcare appointment inquiries."""

    response: str
    urgency_level: str = Field(description="Patient urgency: routine, urgent, emergency")
    appointment_needed: bool
    follow_up_required: bool
    department_referral: Optional[str] = Field(description="Specific medical department if referral needed")


agent2 = Agent(
    model=model,
    output_type=HealthcareResponseModel,
    system_prompt=(
        "You are an intelligent healthcare appointment assistant. "
        "Analyze patient inquiries carefully and provide structured responses. "
        "Always maintain patient confidentiality and professional tone."
    ),
)

# Example: Emergency inquiry with structured response
if __name__ == "__main__":
    response = agent2.run_sync("I have severe chest pain and need to see a cardiologist immediately.")
    print(response.output.model_dump_json(indent=2))


# --------------------------------------------------------------
# 3. Agent with Structured Response & Dependencies
# --------------------------------------------------------------
"""
This example demonstrates how to use dependencies and context in healthcare agents.
Key concepts:
- Defining complex medical data models with Pydantic
- Injecting runtime dependencies
- Using dynamic system prompts
"""


# Define appointment schema
class Appointment(BaseModel):
    """Structure for appointment details."""

    appointment_id: str
    date: date
    time: str
    doctor_name: str
    department: str
    status: str


# Define doctor schema
class Doctor(BaseModel):
    """Structure for doctor information."""

    doctor_id: str
    name: str
    specialty: str
    available_days: List[str]


# Define patient schema
class PatientDetails(BaseModel):
    """Structure for patient information."""

    patient_id: str
    name: str
    email: str
    phone: str
    medical_record_number: str
    appointments: Optional[List[Appointment]] = None
    insurance_provider: Optional[str] = None


# Agent with structured output and dependencies
agent3 = Agent(
    model=model,
    output_type=HealthcareResponseModel,
    deps_type=PatientDetails,
    retries=3,
    system_prompt=(
        "You are an intelligent healthcare appointment assistant. "
        "Analyze patient inquiries carefully and provide structured responses. "
        "Always greet the patient professionally and provide helpful guidance. "
        "Maintain strict patient confidentiality."
    ),
)


# Add dynamic system prompt based on dependencies
@agent3.system_prompt
async def add_patient_context(ctx: RunContext[PatientDetails]) -> str:
    return f"Patient information: {to_markdown(ctx.deps)}"


patient = PatientDetails(
    patient_id="P001",
    name="Jane Doe",
    email="jane.doe@email.com",
    phone="555-0123",
    medical_record_number="MRN-789456",
    insurance_provider="Blue Cross Blue Shield",
    appointments=[
        Appointment(
            appointment_id="A001",
            date=date(2024, 12, 15),
            time="10:00 AM",
            doctor_name="Dr. Smith",
            department="Cardiology",
            status="scheduled"
        ),
    ],
)

# Example: Patient inquiry with context
if __name__ == "__main__":
    response = agent3.run_sync(user_prompt="When is my next appointment?", deps=patient)

    response.all_messages()
    print(response.output.model_dump_json(indent=2))

    print(
        "Patient Details:\n"
        f"Name: {patient.name}\n"
        f"MRN: {patient.medical_record_number}\n"
        f"Insurance: {patient.insurance_provider}\n\n"
        "Response Details:\n"
        f"{response.output.response}\n\n"
        "Assessment:\n"
        f"Urgency Level: {response.output.urgency_level}\n"
        f"Follow-up Required: {response.output.follow_up_required}\n"
        f"Department Referral: {response.output.department_referral}"
    )


# --------------------------------------------------------------
# 4. Agent with Tools
# --------------------------------------------------------------

"""
This example shows how to enhance healthcare agents with custom tools.
Key concepts:
- Creating and registering tools
- Accessing context in tools
- Healthcare-specific tool functions
"""

# Simulated appointment database
appointment_db: Dict[str, Dict] = {
    "A001": {
        "date": "2024-12-15",
        "time": "10:00 AM",
        "doctor": "Dr. Smith",
        "department": "Cardiology",
        "status": "scheduled"
    },
    "A002": {
        "date": "2024-12-20",
        "time": "2:30 PM",
        "doctor": "Dr. Johnson",
        "department": "Dermatology",
        "status": "confirmed"
    },
}

# Doctor availability database
doctor_availability: Dict[str, List[str]] = {
    "Dr. Smith": ["Monday", "Wednesday", "Friday"],
    "Dr. Johnson": ["Tuesday", "Thursday"],
    "Dr. Williams": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
}


def get_appointment_details(ctx: RunContext[PatientDetails]) -> str:
    """Get the patient's appointment details."""
    if ctx.deps.appointments:
        appointment = ctx.deps.appointments[0]
        details = appointment_db.get(appointment.appointment_id, {})
        return f"Appointment {appointment.appointment_id}: {details.get('date')} at {details.get('time')} with {details.get('doctor')} in {details.get('department')}"
    return "No appointments found"


def check_doctor_availability(doctor_name: str) -> str:
    """Check doctor availability."""
    available_days = doctor_availability.get(doctor_name, [])
    if available_days:
        return f"{doctor_name} is available on: {', '.join(available_days)}"
    return f"No availability information found for {doctor_name}"


# Agent with tools
agent4 = Agent(
    model=model,
    output_type=HealthcareResponseModel,
    deps_type=PatientDetails,
    retries=3,
    system_prompt=(
        "You are an intelligent healthcare appointment assistant. "
        "Use tools to look up appointment and doctor information. "
        "Provide accurate, helpful responses while maintaining patient confidentiality. "
        "Always greet the patient professionally."
    ),
    tools=[
        Tool(get_appointment_details, takes_ctx=True),
        Tool(check_doctor_availability, takes_ctx=False)
    ],
)


@agent4.system_prompt
async def add_patient_context(ctx: RunContext[PatientDetails]) -> str:
    return f"Patient information: {to_markdown(ctx.deps)}"


# Example: Using tools for appointment management
if __name__ == "__main__":
    response = agent4.run_sync(
        user_prompt="Can you check when Dr. Smith is available for rescheduling?",
        deps=patient
    )

    response.all_messages()
    print(response.output.model_dump_json(indent=2))

    print(
        "Patient Details:\n"
        f"Name: {patient.name}\n"
        f"MRN: {patient.medical_record_number}\n\n"
        "Response Details:\n"
        f"{response.output.response}\n\n"
        "Assessment:\n"
        f"Urgency Level: {response.output.urgency_level}\n"
        f"Appointment Needed: {response.output.appointment_needed}\n"
        f"Department Referral: {response.output.department_referral}"
    )


# --------------------------------------------------------------
# 5. Agent with Reflection and Self-Correction
# --------------------------------------------------------------

"""
This example demonstrates advanced healthcare agent capabilities with self-correction.
Key concepts:
- Implementing self-reflection for medical accuracy
- Handling errors gracefully with retries
- Using ModelRetry for automatic retries
- Decorator-based tool registration
- Medical validation and safety checks
"""

# Simulated comprehensive appointment database
comprehensive_appointment_db: Dict[str, Dict] = {
    "APT-12345": {
        "date": "2024-12-15",
        "time": "10:00 AM",
        "doctor": "Dr. Smith",
        "department": "Cardiology",
        "status": "scheduled",
        "patient_id": "P001"
    },
    "APT-67890": {
        "date": "2024-12-20",
        "time": "2:30 PM",
        "doctor": "Dr. Johnson",
        "department": "Dermatology",
        "status": "confirmed",
        "patient_id": "P002"
    },
}

patient_validation = PatientDetails(
    patient_id="P001",
    name="Jane Doe",
    email="jane.doe@email.com",
    phone="555-0123",
    medical_record_number="MRN-789456",
)

# Agent with reflection and self-correction
agent5 = Agent(
    model=model,
    output_type=HealthcareResponseModel,
    deps_type=PatientDetails,
    retries=3,
    system_prompt=(
        "You are an intelligent healthcare appointment assistant. "
        "Use tools to look up appointment information accurately. "
        "Always verify appointment IDs and patient information. "
        "Maintain strict patient confidentiality and provide professional responses."
    ),
)


@agent5.tool_plain()
def get_appointment_status(appointment_id: str) -> str:
    """Get the appointment status for a given appointment ID."""
    appointment_info = comprehensive_appointment_db.get(appointment_id)
    if appointment_info is None:
        raise ModelRetry(
            f"No appointment found for ID {appointment_id}. "
            "Please ensure the appointment ID is in the correct format (APT-XXXXX) "
            "and verify with the patient. Self-correct if needed and try again."
        )
    return f"Appointment {appointment_id}: {appointment_info['date']} at {appointment_info['time']} with {appointment_info['doctor']} ({appointment_info['department']}) - Status: {appointment_info['status']}"


@agent5.tool_plain()
def validate_patient_appointment(appointment_id: str, patient_id: str) -> str:
    """Validate that an appointment belongs to the specified patient."""
    appointment_info = comprehensive_appointment_db.get(appointment_id)
    if appointment_info is None:
        raise ModelRetry(f"Appointment {appointment_id} not found. Please verify the appointment ID format.")

    if appointment_info.get('patient_id') != patient_id:
        raise ModelRetry(
            f"Appointment {appointment_id} does not belong to patient {patient_id}. "
            "This is a HIPAA violation. Please verify patient identity before providing appointment information."
        )

    return f"Appointment {appointment_id} verified for patient {patient_id}"


# Example: Self-correction and validation
if __name__ == "__main__":
    # This will trigger self-correction due to incorrect format
    print("\n--- Example 1: Incorrect appointment ID format ---")
    response = agent5.run_sync(
        user_prompt="What's the status of my appointment 12345?",
        deps=patient_validation
    )

    response.all_messages()
    print(response.output.model_dump_json(indent=2))

    # This will work correctly
    print("\n--- Example 2: Correct appointment ID format ---")
    response2 = agent5.run_sync(
        user_prompt="What's the status of my appointment APT-12345?",
        deps=patient_validation
    )

    response2.all_messages()
    print(response2.output.model_dump_json(indent=2))