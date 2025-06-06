import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import re
from dateutil import parser
from datetime import datetime
from geopy.geocoders import Nominatim
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
from datetime import datetime, date

# Plans and programs data
federalPrograms = [
    "SNAP (Food Stamps)",
    "Medicaid",
    "Supplemental Security Income (SSI)",
    "Federal Public Housing Assistance (FPHA)",
    "Veterans Pension and Survivors Benefit",
]

tribalPrograms = [
    "Bureau of Indian Affairs General Assistance",
    "Head Start (income-qualified only)",
    "Tribal TANF",
    "Food Distribution Program on Indian Reservations",
]

plans = [
    # 1) Non-Tribal, incomeQualified=True, programQualified=True
    {"id": 1, "name": "Standard Lifeline Phone & Data Plan", "tribal": False, "incomeQualified": True, "programQualified": True},
    {"id": 5, "name": "Lifeline Unlimited Talk & Text Plan", "tribal": False, "incomeQualified": True, "programQualified": True},
    {"id": 6, "name": "Lifeline Broadband Discount Plan", "tribal": False, "incomeQualified": True, "programQualified": True},

    # 2) Tribal, incomeQualified=True, programQualified=False
    {"id": 2, "name": "Tribal Lifeline Phone & Internet Assistance Plan", "tribal": True, "incomeQualified": True, "programQualified": False},
    {"id": 7, "name": "Tribal Lifeline Prepaid Mobile Plan", "tribal": True, "incomeQualified": True, "programQualified": False},
    {"id": 8, "name": "Tribal Emergency Lifeline Phone Plan", "tribal": True, "incomeQualified": True, "programQualified": False},

    # 3) Tribal, incomeQualified=False, programQualified=True
    {"id": 3, "name": "Native American Lifeline Broadband & Mobile Plan", "tribal": True, "incomeQualified": False, "programQualified": True},
    {"id": 9, "name": "Tribal Lifeline Family Internet Plan", "tribal": True, "incomeQualified": False, "programQualified": True},
    {"id": 10, "name": "Tribal Lifeline Senior Citizen Phone Plan", "tribal": True, "incomeQualified": False, "programQualified": True},

    # 4) Non-Tribal, incomeQualified=True, programQualified=False
    {"id": 4, "name": "Low-Income Lifeline Wireless Service Plan", "tribal": False, "incomeQualified": True, "programQualified": False},
    {"id": 11, "name": "Affordable Lifeline Data & Voice Plan", "tribal": False, "incomeQualified": True, "programQualified": False},
    {"id": 12, "name": "Basic Lifeline Phone Service Plan", "tribal": False, "incomeQualified": True, "programQualified": False},
]

# Initialize session state variables with defaults if missing
def init_session_state():
    defaults = {
        "ocr_reader": easyocr.Reader(['en'], gpu=False),
        "step": 1,
        "gov_id_file": None,
        "id_verified": False,
        "id_verification_error": "",
        "user_details": {"name": "", "address": "", "dob": ""},
        "tribal_region": False,
        "tribal_land_verified": None,
        "tribal_land_error": "",
        "latitude": None,
        "longitude": None,
        "tribal_id_or_ssn": "",
        "is_applying_self": True,
        "bqp_details": {"name": "", "address": "", "dob": ""},
        "qualification_type": "income",
        "selected_programs": [],
        "fast_track": False,
        "available_plans": [],
        "selected_plan_id": None,
        "nlad_verified": None,
        "nlad_error": "",
        "confirmed": False,
        "enrolled": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

def verify_ocr_and_extract_details(uploaded_file):
    reader = st.session_state.ocr_reader
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)[:, :, ::-1]
    results = reader.readtext(img_np)

    st.write("🧾 OCR detected text blocks:")
    for res in results:
        st.write(res[1])

    extracted_name = ""
    extracted_address = ""
    extracted_dob = ""

    # Flatten OCR results into one string for better pattern recognition
    full_text = "\n".join([res[1] for res in results]).lower()

    # --- Extract DOB ---
    dob_match = re.search(r'(dob|birth|date of birth)[:\s\-]*([0-9]{2,4}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{2,4})', full_text)
    if dob_match:
        extracted_dob = dob_match.group(2).strip()

    # --- Extract Name ---
    # Look for patterns like LAST, FIRST MIDDLE
    name_match = re.search(r'\b([a-z]+),\s*([a-z]+)(\s+[a-z]+)?\b', full_text)
    if name_match:
        extracted_name = f"{name_match.group(2).strip().title()} {name_match.group(3).strip().title() if name_match.group(3) else ''} {name_match.group(1).strip().title()}".strip()

    # --- Extract Address ---
    # Look for number + street + city/state line
    address_match = re.search(r'(\d{3,5}\s+[a-z0-9\s]+(?:st|ave|rd|dr|blvd|ln|ct)\.?\s*\n?.+?\n?.*?\d{5})', full_text)
    if address_match:
        extracted_address = address_match.group(1).strip().title()

    # Store in session state
    st.session_state.user_details = {
        "name": extracted_name,
        "address": extracted_address,
        "dob": extracted_dob
    }

    st.session_state.id_verified = True
    st.session_state.id_verification_error = ""



GEOJSON_URL = "https://opendata.arcgis.com/datasets/4a433ecfceee4ad5a6f1a96d35ff22a0_0.geojson"

def geocode_address(address, retries=3, delay=2):
    geolocator = Nominatim(user_agent="streamlit_acp_app")
    for attempt in range(retries):
        try:
            location = geolocator.geocode(address, timeout=5)
            if location:
                return location.latitude, location.longitude
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error(f"Geocoding failed after {retries} attempts: {e}")
    return None, None

def is_point_in_tribal_area(latitude, longitude, geojson_url=GEOJSON_URL):
    # Load GeoJSON polygons once (you can optimize by caching)
    gdf = gpd.read_file(geojson_url)

    point = Point(longitude, latitude)  # Note order: lon, lat
    return gdf.contains(point).any()

def display_map(latitude, longitude):
    if latitude is None or longitude is None:
        st.error("Could not determine location from address.")
        return

    data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

    fig = px.scatter_mapbox(
        data,
        lat='lat',
        lon='lon',
        zoom=15,
        height=300,
        width=None,
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig, use_container_width=True)

def display_tribal_map_with_point(latitude, longitude, gdf):
    geojson = gdf.__geo_interface__

    fig = go.Figure()

    # Tribal boundary layer
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        locations=gdf.index,
        z=[1] * len(gdf),
        colorscale="Viridis",
        showscale=False,
        marker_opacity=0.3,
        marker_line_width=1,
        name="Tribal Boundaries",
    ))

    # User location marker
    fig.add_trace(go.Scattermapbox(
        lat=[latitude],
        lon=[longitude],
        mode='markers+text',
        text=["Your Location"],
        textposition="top center",
        marker=go.scattermapbox.Marker(size=14, color='red'),
        name="Your Location"
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=10,
        mapbox_center={"lat": latitude, "lon": longitude},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def filter_plans():
    # Filter plans by tribal region and qualification
    tribal = st.session_state.tribal_region
    qual_type = st.session_state.qualification_type
    selected_programs = st.session_state.selected_programs

    filtered = []
    for plan in plans:
        if tribal and not plan["tribal"]:
            continue
        if not tribal and plan["tribal"]:
            continue

        if qual_type == "income":
            if plan["incomeQualified"]:
                filtered.append(plan)
        else:  # program
            # Check if any selected program qualifies for plan
            if plan["programQualified"]:
                # Simplify: assume if any program selected, plan qualifies
                if len(selected_programs) > 0:
                    filtered.append(plan)

    st.session_state.available_plans = filtered

def verify_nlad():
    # Dummy simulation - In real app, call backend API or service
    st.session_state.nlad_verified = True
    st.session_state.nlad_error = ""

def enroll_user():
    # Dummy simulation for enrollment
    st.session_state.enrolled = True

def next_step():
    st.session_state.step += 1

def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1

st.title("ACP/Lifeline Application")

if st.session_state.step == 1:
    st.header("Step 1: Upload Your ID")
    uploaded_file = st.file_uploader("Upload your government-issued ID", type=["png", "jpg", "jpeg","jfif"])

    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Uploaded ID", use_container_width=True)
        st.session_state.gov_id_file = uploaded_file

        if st.button("Extract Details via OCR"):
            verify_ocr_and_extract_details(uploaded_file)

    # ✅ Show extracted details
    if st.session_state.get("id_verified"):
        st.success("OCR complete. Please review and edit your details if needed:")

        details = st.session_state.user_details
        st.session_state.user_details["name"] = st.text_input("Name", value=details.get("name", ""))
        st.session_state.user_details["address"] = st.text_area("Address", value=details.get("address", ""))
        st.session_state.user_details["dob"] = st.text_input("Date of Birth", value=details.get("dob", ""))

        if all(v == "" for v in details.values()):
            st.warning("We couldn't extract any details. Please fill them in manually.")

# Trigger reset via a session flag
if st.button("Reset and Re-upload"):
    st.session_state.reset_triggered = True

# Perform reset outside the button block
if st.session_state.get("reset_triggered"):
    st.session_state.pop("id_verified", None)
    st.session_state.pop("user_details", None)
    st.session_state.pop("reset_triggered", None)
    st.experimental_rerun()

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", disabled=True)
    with col2:
        st.button("Next", disabled=not st.session_state.get("id_verified", False), on_click=next_step)



# -------- Step 2 --------
elif st.session_state.step == 2:
    st.header("Step 2: Confirm or Edit Your Details")

    st.session_state.user_details["name"] = st.text_input(
        "Full Name", value=st.session_state.user_details["name"]
    )
    st.session_state.user_details["address"] = st.text_input(
        "Address", value=st.session_state.user_details["address"]
    )

    dob_str = st.session_state.user_details.get("dob", "")
    dob_date = None
    if dob_str:
        try:
            dob_date = parser.parse(dob_str).date()
        except:
            dob_date = datetime.today().date()
    else:
        dob_date = datetime.today().date()

    dob_selected = st.date_input("Date of Birth", value=dob_date)
    st.session_state.user_details["dob"] = dob_selected.strftime("%Y-%m-%d")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_step)
    with col2:
        disabled = (
            st.session_state.user_details["name"] == ""
            or st.session_state.user_details["address"] == ""
            or st.session_state.user_details["dob"] == ""
        )
        st.button("Next", disabled=disabled, on_click=next_step)

# -------- Step 3 --------
elif st.session_state.step == 3:
    st.header("Step 3: Are You in a Tribal Region?")

    # Capture and store choice
    tribal_region = st.radio(
        "Select tribal region status:",
        options=[False, True],
        format_func=lambda x: "Yes (Tribal Region)" if x else "No",
        index=1 if st.session_state.get("tribal_region", False) else 0,
    )
    st.session_state.tribal_region = tribal_region

    def go_forward_from_step3():
        if st.session_state.tribal_region:
            st.session_state.step = 4
        else:
            st.session_state.step = 5

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_step)
    with col2:
        st.button("Next", on_click=go_forward_from_step3)



# -------- Step 4 --------
if st.session_state.step == 4:
    st.header("Step 4: Verify Tribal Land Address")

    address = st.session_state.user_details.get("address", "")
    if address:
        st.write(f"Verifying address: {address}")
        lat, lon = geocode_address(address)
        st.session_state.latitude = lat
        st.session_state.longitude = lon

        if lat and lon:
            display_map(lat, lon)
            st.success("Address location displayed on map.")
            verified = st.checkbox("Confirm your address is within Tribal Lands", value=True)
            st.session_state.tribal_land_verified = verified
        else:
            st.error("Could not geocode the address.")
            st.session_state.tribal_land_verified = False
    else:
        st.error("No address provided.")
        st.session_state.tribal_land_verified = False

    def go_forward_from_step4():
        if st.session_state.get("tribal_land_verified"):
            st.session_state.step = 5

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_step)
    with col2:
        st.button(
            "Next",
            disabled=not st.session_state.get("tribal_land_verified", False),
            on_click=go_forward_from_step4,
        )

# -------- Step 5 --------
elif st.session_state.step == 5:
    if st.session_state.tribal_region:
        st.header("Step 5: Enter Tribal ID or Social Security Number")
        label = "Tribal ID or Last 4 digits of SSN"
    else:
        st.header("Step 5: Enter Your SSN")
        label = "Last 4 digits of SSN"

    input_value = st.text_input(label, value=st.session_state.get("tribal_id_or_ssn", ""))
    st.session_state.tribal_id_or_ssn = input_value.strip()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.step = 4 if st.session_state.tribal_region else 3
    with col2:
        st.button(
            "Next",
            disabled=(not input_value.strip().isdigit() or len(input_value.strip()) != 4),
            on_click=next_step,
        )


# -------- Step 6 --------
elif st.session_state.step == 6:
    st.header("Step 6: Are You Applying for Yourself or Someone Else?")

    # Radio button to choose applicant
    applying_self = st.radio(
        "Select who is applying:",
        options=["I am applying for myself", "I am applying for someone else"],
        index=0 if st.session_state.is_applying_self else 1
    )
    st.session_state.is_applying_self = (applying_self == "I am applying for myself")

    # If applying for someone else, show extra fields
    if not st.session_state.is_applying_self:
        st.write("Please provide details for the person you are applying for:")

        # Name and address input
        st.session_state.bqp_details["name"] = st.text_input(
            "Full Name", value=st.session_state.bqp_details.get("name", "")
        )
        st.session_state.bqp_details["address"] = st.text_input(
            "Address", value=st.session_state.bqp_details.get("address", "")
        )

        # Date of Birth
        dob_str = st.session_state.bqp_details.get("dob", "")
        try:
            dob_date = datetime.strptime(dob_str, "%Y-%m-%d").date()
        except:
            dob_date = date(2000, 1, 1)  # default fallback

        dob_selected = st.date_input(
            "Date of Birth",
            value=dob_date,
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            key="bqp_dob"
        )
        st.session_state.bqp_details["dob"] = dob_selected.strftime("%Y-%m-%d")

        # SSN or Tribal ID input
        st.write("Enter one of the following for identity verification:")
        st.session_state.bqp_details["ssn"] = st.text_input(
            "Last 4 digits of SSN", value=st.session_state.bqp_details.get("ssn", "")
        )
        st.session_state.bqp_details["tribal_id"] = st.text_input(
            "Tribal ID (if applicable)", value=st.session_state.bqp_details.get("tribal_id", "")
        )

    # Navigation Buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_step)
    with col2:
        # Disable Next button if required fields are missing (when not self)
        if not st.session_state.is_applying_self:
            bqp = st.session_state.bqp_details
            disabled = (
                not bqp.get("name") or
                not bqp.get("address") or
                not bqp.get("dob") or
                (not bqp.get("ssn") and not bqp.get("tribal_id"))
            )
        else:
            disabled = False

        st.button("Next", disabled=disabled, on_click=next_step)


# -------- Step 7 --------
elif st.session_state.step == 7:
    st.header("Step 7: Select Qualification Type")

    qualification_type = st.radio(
        "Do you qualify based on income or government program?",
        options=["income", "program"],
        index=0 if st.session_state.qualification_type != "program" else 1,
    )
    st.session_state.qualification_type = qualification_type

    if qualification_type == "program":
        selected_programs = st.multiselect(
            "Select the programs you participate in:",
            federalPrograms + tribalPrograms,
            default=st.session_state.selected_programs,
        )
        st.session_state.selected_programs = selected_programs
    else:
        st.subheader("Income-Based Qualification")

        state = st.selectbox("Select your state/territory:", ["48 Contiguous States", "Alaska", "Hawaii"])
        household_size = st.number_input("Household Size", min_value=1, max_value=20, step=1)
        income = st.number_input("Your Annual Household Income ($)", min_value=0, step=100)

        st.session_state.income_info = {
            "state": state,
            "household_size": household_size,
            "income": income
        }

        # 2025 Poverty Guidelines at 135%
        guidelines = {
            "48 Contiguous States": [21128, 28553, 35978, 43403, 50828, 58253, 65678, 73103],
            "Alaska": [26393, 35681, 44969, 54257, 63545, 72833, 82121, 91409],
            "Hawaii": [24287, 32832, 41378, 49923, 58469, 67041, 75560, 84105]
        }

        base = guidelines[state]
        if household_size <= 8:
            limit = base[household_size - 1]
        else:
            extra = {
                "48 Contiguous States": 7425,
                "Alaska": 9288,
                "Hawaii": 8546
            }[state]
            limit = base[-1] + (household_size - 8) * extra

        qualifies = income <= limit
        st.markdown(f"**135% Federal Poverty Limit for your case: ${limit:,}**")
        st.markdown(f"**You {'qualify ✅' if qualifies else 'do not qualify ❌'} based on income.**")
        st.session_state.income_qualified = qualifies

    # Navigation Buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_step)
    with col2:
        if qualification_type == "program":
            disabled = len(st.session_state.selected_programs) == 0
        else:
            income_info = st.session_state.get("income_info", {})
            disabled = income_info.get("income", 0) == 0 or not st.session_state.get("income_qualified", False)

        st.button("Next", disabled=disabled, on_click=next_step)

# -------- Step 8 --------
elif st.session_state.step == 8:
    st.header("Step 8: Application Processing Options")
    fast_track = st.radio(
        "Would you like to fast track your application?",
        options=[True, False],
        index=0 if st.session_state.fast_track else 1,
    )
    st.session_state.fast_track = fast_track

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_step)
    with col2:
        st.button("Next", on_click=next_step)

# -------- Step 9 --------
elif st.session_state.step == 9:
    st.header("Step 9: Available Plans")
    if not st.session_state.available_plans:
        filter_plans()

    if len(st.session_state.available_plans) == 0:
        st.warning("No plans available for your qualification and region.")
    else:
        plan_names = [p["name"] for p in st.session_state.available_plans]
        selected = st.radio("Select a plan:", options=plan_names)
        for plan in st.session_state.available_plans:
            if plan["name"] == selected:
                st.session_state.selected_plan_id = plan["id"]

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=prev_step)
    with col2:
        st.button("Next", disabled=(st.session_state.selected_plan_id is None), on_click=next_step)

# -------- Step 10 --------
elif st.session_state.step == 10:
    st.header("Step 10: NLAD Verification")

    if st.session_state.nlad_verified is None:
        with st.spinner("Verifying against NLAD..."):
            verify_nlad()

    if st.session_state.nlad_verified:
        st.success("NLAD verification successful.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Back", on_click=prev_step)
        with col2:
            st.button("Next", on_click=next_step)
    else:
        st.error(st.session_state.nlad_error or "NLAD verification failed.")
        st.text("Please contact support or provide additional documentation.")
        st.button("Back", on_click=prev_step)

# -------- Step 11 --------
elif st.session_state.step == 11:
    st.header("Step 11: Confirm and Submit Your Application")
    st.write("Please review all your details carefully before submitting.")

    st.subheader("Personal Details")
    if st.session_state.is_applying_self:
        st.write(f"Name: {st.session_state.user_details['name']}")
        st.write(f"Address: {st.session_state.user_details['address']}")
        st.write(f"Date of Birth: {st.session_state.user_details['dob']}")
    else:
        st.write(f"Name (BQP): {st.session_state.bqp_details['name']}")
        st.write(f"Address (BQP): {st.session_state.bqp_details['address']}")
        st.write(f"Date of Birth (BQP): {st.session_state.bqp_details['dob']}")

    st.subheader("Qualification")
    st.write(f"Qualification Type: {st.session_state.qualification_type.title()}")
    if st.session_state.qualification_type == "program":
        st.write(f"Programs: {', '.join(st.session_state.selected_programs)}")

    st.subheader("Selected Plan")
    selected_plan = next(
        (p for p in st.session_state.available_plans if p["id"] == st.session_state.selected_plan_id), None
    )
    if selected_plan:
        st.write(selected_plan["name"])

    if not st.session_state.confirmed:
        if st.button("Confirm and Submit"):
            st.session_state.confirmed = True
            enroll_user()
            st.success("Application submitted successfully! You are now enrolled.")
    else:
        if st.session_state.enrolled:
            st.balloons()
            st.success("You have been successfully enrolled. Thank you!")
        else:
            st.info("Submitting your application...")

    st.button("Back", on_click=prev_step)


