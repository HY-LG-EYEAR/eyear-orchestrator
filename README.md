# EYEAR Orchestrator

Role-based agents and central orchestration for [EYEAR](https://github.com/HY-LG-EYEAR).

**Maintainer:** Jungbin Song

Cleaning, Noise, Temperature, and Lighting agents use the World Model to propose actions. The Home Orchestrator coordinates their proposals through the OpenAI API, validates decisions, and sends commands to the simulated appliances.

This module also manages sleep-intent questions, direct voice commands, and decision records. World Model data and execution feedback are its main inputs.

**Stack:** Python 3.11, Pydantic, OpenAI API.
