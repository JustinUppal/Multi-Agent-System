"""The client persona the simulator agent role-plays."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClientProfile:
    """A retail client's financial situation, roughly an intake form."""

    name: str
    age: int
    occupation: str
    risk_aversion: str  # "conservative" | "moderate" | "aggressive"
    annual_income_usd: int
    liquid_assets_usd: int
    current_investments: dict[str, int]  # instrument -> approx USD value
    goals: list[str] = field(default_factory=list)
    time_horizon_years: int = 10

    def to_prompt(self) -> str:
        """Render the profile as a second-person briefing for the client agent."""
        investments = "\n".join(
            f"  - {name}: ${value:,}" for name, value in self.current_investments.items()
        ) or "  - (none)"
        goals = "\n".join(f"  - {goal}" for goal in self.goals) or "  - (none stated)"
        return (
            f"You are {self.name}, a {self.age}-year-old {self.occupation}.\n"
            f"Risk attitude: {self.risk_aversion}.\n"
            f"Annual income: ${self.annual_income_usd:,}.\n"
            f"Investable cash on hand: ${self.liquid_assets_usd:,}.\n"
            f"Investment time horizon: {self.time_horizon_years} years.\n"
            f"Current investments:\n{investments}\n"
            f"Financial goals:\n{goals}"
        )


# Sample client for the demo. Dummy data, loosely based on me.
SAMPLE_CLIENT = ClientProfile(
    name="Justin Uppal",
    age=29,
    occupation="software engineer",
    risk_aversion="moderate",
    annual_income_usd=180_000,
    liquid_assets_usd=120_000,
    current_investments={
        "401(k) target-date fund": 210_000,
        "Company RSUs (single tech stock)": 95_000,
        "High-yield savings": 40_000,
    },
    goals=[
        "Put the $120k cash to work without taking on reckless risk",
        "Reduce over-concentration in my employer's stock",
        "Be on track to retire comfortably around age 60",
    ],
    time_horizon_years=31,
)
