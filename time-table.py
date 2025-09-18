from __future__ import annotations

import io
import itertools
import json
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

DAYS_DEFAULT = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

PERIODS = [
    ("P1", "07:15", "08:00"),
    ("P2", "08:00", "08:45"),
    ("P3", "08:45", "09:30"),
    # Break 09:30–10:00 (not a schedulable slot)
    ("P4", "10:00", "10:45"),
    ("P5", "10:45", "11:30"),
    ("P6", "11:30", "12:15"),
]

# Modified: Activity and Study are now included as major subjects
CORE_SUBJECTS = [
    "Hindi",
    "Gujarati",
    "Science",
    "Social Science",
    "English",
    "Maths",       
    "Sanskrut",
    "Activity",    # Added as major subject
    "Study",       # Added as major subject
]

# Modified: Added faculty assignments for Activity and Study
TEACHERS_BY_SUBJECT: Dict[str, List[str]] = {
    "Hindi": ["Dhruv Rakholiya","Dhruv Vachani"],
    "Gujarati": ["Meet Navadiya","Meet Kankotiya"],
    "Science": ["Ravi Ramani","Dhruv Vaghasiya"],
    "Social Science": ["Jenil Bhayani","Rakesh Patel"],
    "English": ["Yash Sakariya","Dhruvit kotadiya"],
    "Maths": ["Jaydeep Patel","Krupal Dhameliya"],
    "Sanskrut": ["Het Sheladiya","kaushik khatri"],
    "Activity": ["Mohit Ramani","Tejas Patel"],    # Added faculty for Activity
    "Study": ["Ujas Desai","Akshay Shekhda"],        # Added faculty for Study
}

CLASSES = [f"Std{std}-{sec}" for std in range(1, 6) for sec in ["A", "B"]]
ROOMS = list(range(101, 111))
CLASS_TO_ROOM = {c: room for c, room in zip(CLASSES, ROOMS)}

# Removed these constants as Activity and Study are now regular subjects
# ACTIVITY_NAME = "Activity"  
# FILLER_NAME = "Study"

@dataclass
class ProblemSpec:
    classes: List[str]
    class_to_room: Dict[str, int]
    days: List[str]
    periods: List[Tuple[str, str, str]]  
    core_subjects: List[str]
    weekly_quota: Dict[str, int]         
    # Removed fill_with_activity parameter as it's no longer needed

    @property
    def total_slots_per_class(self) -> int:
        return len(self.days) * len(self.periods)

    def validate(self) -> Tuple[bool, str]:
        requested = sum(self.weekly_quota.values())
        total = self.total_slots_per_class
        if requested > total:
            return False, (
                f"Requested weekly total ({requested}) exceeds available slots ({total}).\n"
                f"Lower some subject quotas or reduce active days/periods."
            )
        if requested < total:
            return False, (
                f"Requested weekly total ({requested}) is less than available slots ({total}).\n"
                f"Increase some subject quotas to fill all time slots."
            )
        return True, "OK"


def solve_timetable(spec: ProblemSpec):
    """Return {class_name: ndarray[periods x days] with subject (and teacher) strings} or raise ValueError."""
    ok, msg = spec.validate()
    if not ok:
        raise ValueError(msg)

    # Feasibility pre-check for teacher capacity
    total_time_slots = len(spec.days) * len(spec.periods)
    classes_count = len(spec.classes)
    for s_name in spec.core_subjects:
        required = classes_count * spec.weekly_quota.get(s_name, 0)
        capacity = total_time_slots * len(TEACHERS_BY_SUBJECT.get(s_name, []))
        if required > capacity:
            raise ValueError(
                f"Teacher capacity infeasible for subject '{s_name}': required {required} sessions/week across all classes, "
                f"but available capacity is {capacity} (days*periods*teachers). Add more teachers for this subject or lower quotas."
            )

    model = cp_model.CpModel()

    classes = spec.classes
    days = spec.days
    periods = spec.periods
    subjects = list(spec.core_subjects)  # All subjects are now treated equally

    C, D, P, S = len(classes), len(days), len(periods), len(subjects)

    def var_name(ci, di, pi, si):
        return f"x_c{ci}_d{di}_p{pi}_s{si}"

    x = {}
    for ci in range(C):
        for di in range(D):
            for pi in range(P):
                for si in range(S):
                    x[(ci, di, pi, si)] = model.NewBoolVar(var_name(ci, di, pi, si))

    # Teacher assignment variables
    def t_var_name(ci, di, pi, teacher):
        return f"t_c{ci}d{di}_p{pi}_t{teacher}"

    t = {}
    for ci in range(C):
        for di in range(D):
            for pi in range(P):
                for s_name in subjects:
                    for teacher in TEACHERS_BY_SUBJECT.get(s_name, []):
                        t[(ci, di, pi, teacher)] = model.NewBoolVar(t_var_name(ci, di, pi, teacher))

    # Constraint: Each slot must have exactly one subject
    for ci in range(C):
        for di in range(D):
            for pi in range(P):
                model.Add(sum(x[(ci, di, pi, si)] for si in range(S)) == 1)

    # Constraint: At most one session per subject per day per class
    for ci in range(C):
        for di in range(D):
            for si in range(S):
                model.Add(sum(x[(ci, di, pi, si)] for pi in range(P)) <= 1)

    # Constraint: No consecutive periods for the same subject
    for ci in range(C):
        for di in range(D):
            for si in range(S):
                for pi in range(P - 1):
                    model.Add(x[(ci, di, pi, si)] + x[(ci, di, pi + 1, si)] <= 1)

    # Constraint: Weekly quota for each subject
    for ci in range(C):
        for s_name in subjects:
            s_idx = subjects.index(s_name)
            quota = spec.weekly_quota.get(s_name, 0)
            model.Add(sum(x[(ci, di, pi, s_idx)] for di in range(D) for pi in range(P)) == quota)

    # Link teacher assignment to subject selection
    for ci in range(C):
        for di in range(D):
            for pi in range(P):
                for s_name in subjects:
                    s_idx = subjects.index(s_name)
                    teachers_for_s = TEACHERS_BY_SUBJECT.get(s_name, [])
                    if teachers_for_s:
                        model.Add(
                            sum(t[(ci, di, pi, teacher)] for teacher in teachers_for_s) == x[(ci, di, pi, s_idx)]
                        )
                    else:
                        model.Add(x[(ci, di, pi, s_idx)] == 0)

    # Constraint: A teacher cannot teach two classes at the same time
    all_teachers = sorted({teacher for lst in TEACHERS_BY_SUBJECT.values() for teacher in lst})
    for di in range(D):
        for pi in range(P):
            for teacher in all_teachers:
                model.Add(sum(t.get((ci, di, pi, teacher), 0) for ci in range(C)) <= 1)

    # Objective: Prefer scheduling core subjects in early periods
    early_periods = list(range(min(3, P)))
    obj_terms = []
    for ci in range(C):
        for di in range(D):
            for s_name in subjects:
                s_idx = subjects.index(s_name)
                for pi in early_periods:
                    obj_terms.append(x[(ci, di, pi, s_idx)])
    if obj_terms:
        model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError("No feasible schedule found with current constraints.")

    schedules: Dict[str, np.ndarray] = {}
    for ci, cls in enumerate(classes):
        grid = np.empty((P, D), dtype=object)
        for di, _day in enumerate(days):
            for pi, _per in enumerate(periods):
                for si, subj in enumerate(subjects):
                    if solver.Value(x[(ci, di, pi, si)]) == 1:
                        # Determine teacher for this subject-slot
                        teacher_suffix = ""
                        if subj in TEACHERS_BY_SUBJECT:
                            assigned = None
                            for teacher in TEACHERS_BY_SUBJECT[subj]:
                                if solver.Value(t[(ci, di, pi, teacher)]) == 1:
                                    assigned = teacher
                                    break
                            if assigned:
                                teacher_suffix = f"\n({assigned})"
                        grid[pi, di] = f"{subj}{teacher_suffix}"
                        break
        schedules[cls] = grid

    return schedules, subjects


def schedule_to_dataframe(days: List[str], periods: List[Tuple[str, str, str]], grid: np.ndarray) -> pd.DataFrame:
    idx = [f"{code}  {start}–{end}" for code, start, end in periods]
    df = pd.DataFrame(grid, index=idx, columns=days)
    
    break_row = pd.DataFrame([["BREAK"] * len(days)], 
                            index=["BREAK  09:30–10:00"], 
                            columns=days)
    
    df_top = df.iloc[:3]  
    df_bottom = df.iloc[3:]  
    
    result_df = pd.concat([df_top, break_row, df_bottom])
    
    return result_df


def build_default_quota(subjects: List[str], default_each: int = 4) -> Dict[str, int]:
    return {s: default_each for s in subjects}


st.set_page_config(
    page_title="School Timetable Generator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 School Timetable Generator")

with st.sidebar:
    st.markdown('<div style="height: calc(100vh - 100px);"></div>', unsafe_allow_html=True)
    generate = st.button("Generate Timetables", type="primary", use_container_width=True)

# Modified: Updated quotas to include Activity and Study with specific allocations
quotas: Dict[str, int] = {
    "Hindi": 4,
    "Gujarati": 4,
    "Science": 4,
    "Social Science": 4,
    "English": 4,
    "Maths": 4,
    "Sanskrut": 4,
    "Activity": 4,  # 4 sessions per week for Activity
    "Study": 4,     # 4 sessions per week for Study
}

if generate:
    try:
        spec = ProblemSpec(
            classes=CLASSES,
            class_to_room=CLASS_TO_ROOM,
            days=DAYS_DEFAULT,
            periods=PERIODS,
            core_subjects=CORE_SUBJECTS,
            weekly_quota=quotas,
            # Removed fill_with_activity parameter
        )
        schedules, subjects_all = solve_timetable(spec)

        st.success("Timetables generated successfully!")

        class_tabs = st.tabs([f"{cls} (Room {CLASS_TO_ROOM[cls]})" for cls in CLASSES])

        for tab, cls in zip(class_tabs, CLASSES):
            with tab:
                df = schedule_to_dataframe(DAYS_DEFAULT, PERIODS, schedules[cls])
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv().encode("utf-8")
                st.download_button(
                    label=f"Download {cls} CSV",
                    data=csv,
                    file_name=f"{cls.replace(' ', '_')}_timetable.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # Keep only the JSON download option
        export_json = {
            cls: schedule_to_dataframe(DAYS_DEFAULT, PERIODS, schedules[cls]).to_dict()
            for cls in CLASSES
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_json, indent=2).encode("utf-8"),
            file_name="timetables.json",
            mime="application/json",
            use_container_width=True,
        )

        total_slots = len(DAYS_DEFAULT) * len(PERIODS)
        with st.expander("Scheduling summary / diagnostics"):
            unmapped = [s for s in CORE_SUBJECTS if not TEACHERS_BY_SUBJECT.get(s)]
            st.write({
                "classes": CLASSES,
                "rooms": CLASS_TO_ROOM,
                "subjects": CORE_SUBJECTS,
                "teachers_by_subject": TEACHERS_BY_SUBJECT,
                "weekly_quota": quotas,
                "per_class_total_slots": total_slots,
                "unmapped_subjects_for_faculty": unmapped,
            })
            if unmapped:
                st.warning("Some subjects do not have any teachers assigned. Please edit TEACHERS_BY_SUBJECT in the code.")
            st.caption("If you see infeasibility, adjust subject quotas to match total available slots.")

    except ValueError as e:
        st.error(str(e))

else:
    st.info("Click 'Generate Timetables' to create schedules for all classes.")
    
    # Display current configuration
    with st.expander("Current Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Subjects & Weekly Quotas")
            for subject, quota in quotas.items():
                teacher = TEACHERS_BY_SUBJECT.get(subject, ["Not Assigned"])[0]
                st.write(f"**{subject}**: {quota} sessions/week ({teacher})")
        
        with col2:
            st.subheader("Schedule Details")
            st.write(f"**Classes**: {len(CLASSES)} classes")
            st.write(f"**Days**: {len(DAYS_DEFAULT)} days")
            st.write(f"**Periods per day**: {len(PERIODS)} periods")
            st.write(f"**Total slots per class**: {len(DAYS_DEFAULT) * len(PERIODS)} slots")
            st.write(f"**Total weekly sessions**: {sum(quotas.values())} sessions")
