import json
import re

from flusk_app.ocr.google_ocr import GoogleVisionOCR


def extract_text(b64_str: str):
    ocr = GoogleVisionOCR("AIzaSyBBcVCeQDVulbk7c_eq-PThNF0TlSLAHCk")
    text = ocr.detect_text(b64_str)

    is_old_nic = False
    nic_no = get_old_nic_no(text)

    is_new_nic = check_is_new_nic(text)
    if is_new_nic:
        nic_no = get_new_nic_no(text)

    driving_licence_no = None
    is_driving_licence = check_is_driving_licence(text)
    if is_driving_licence:
        driving_licence_no = get_driving_licence_no(text)
        nic_no = get_old_nic_no(text)
        if nic_no is None:
            nic_no = get_new_nic_no(text)

    is_passport = check_is_passport(text)
    passport_no = None
    if is_passport:
        passport_no = get_passport_no(text)

    if not is_driving_licence:
        if not is_new_nic:
            if not is_passport:
                if nic_no is not None:
                    is_old_nic = True

    print("is_new_nic", is_new_nic)
    print("is_driving_licence", is_driving_licence)
    print("is_passport", is_passport)
    print("is_old_nic", is_old_nic)

    print("nic_no", nic_no)
    print("driving_licence_no", driving_licence_no)
    print("passport_no", passport_no)

    data = {
        "is_old_nic": is_old_nic,
        "is_new_nic": is_new_nic,
        "is_driving_licence": is_driving_licence,
        "is_passport": is_passport,
        "nic_no": nic_no,
        "driving_licence_no": driving_licence_no,
        "passport_no": passport_no,
    }

    #result = {k: v for k, v in data.items() if v is not None}

    return data


def get_passport_no(text: str):
    up = text.upper()
    m = re.search(r'([A-Z]\s*(?:\d\s*){8})(?!\d)', up)
    return re.sub(r'\s+', '', m.group(1)) if m else None


def check_is_passport(text: str) -> bool:
    return bool(re.search(r"\bPASSPORT\b", text.upper()))


def get_driving_licence_no(text: str):
    up = text.upper()
    # B + 7 digits, spaces allowed
    m = re.search(r'B\s*\d(?:\s*\d){7}', up)
    if not m:
        return None
    # Remove any spaces inside the matched token
    return re.sub(r'\s+', '', m.group(0))


def check_is_driving_licence(text: str) -> bool:
    up = text.upper()
    return all(re.search(rf"\b{w}\b", up) for w in ("DRIVING", "LICENCE"))


def get_new_nic_no(text: str):
    m = re.search(r'(?<!\d)(\d{12})(?!\d)', text)
    return m.group(1) if m else None


def check_is_new_nic(text: str) -> bool:
    up = text.upper()
    return all(re.search(rf"\b{w}\b", up) for w in ("NATIONAL", "IDENTITY", "CARD"))


def get_old_nic_no(txt: str):
    normalized = re.sub(r'[^A-Z0-9]', '', txt.upper())
    m = re.search(r'(?<!\d)(\d{9})V', normalized)
    return (m.group(1) + 'V') if m else None
