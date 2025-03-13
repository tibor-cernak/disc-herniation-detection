from collections import defaultdict
import os
import re
import pydicom

# Typing
from typing import Dict, List, Tuple, Set, Optional
from pydicom.dataset import FileDataset

from series import Series
from annotated_series import AnnotatedSeries
from annotation import Annotation
from mri_type import MRIType

from codes import TH_LS_CODES, C_TH_CODES

import shutil

from typing import List, Tuple


class Study:
    """Represents an MRI study containing multiple DICOM series with annotations."""

    def __init__(self, directory_path: str):
        """
        Initializes the Study object by loading DICOM files, extracting series, and annotations.

        Args:
            directory_path (str): Path to the directory containing the study's files.
        """
        self.directory_path: str = directory_path
        self.directory: List[str] = os.listdir(directory_path)
        self.dicom_files: List[FileDataset] = self.get_dicom_files()
        if len(self.dicom_files) == 0:
            print("No DICOM files in study. Moving to bad_data directory")
            self.move_bad_data(reason="no_files")
            return

        self.study_uid: str = self.get_study_uid()
        self.series: List[Series] = self.group_dicom_files_by_series()
        self.study_text: Optional[str] = self.get_study_text()

        if not self.study_text:
            print(f"Study text not found. Moving to bad_data directory.")
            self.move_bad_data(reason="no_text_file")
            return

        self.annotated_series: List[AnnotatedSeries] = self.get_annotated_series()
        for series in self.annotated_series:
            if len(series.annotations) == 0:
                print("No annotations for the series. Moving to bad_data directory")
                self.move_bad_data(reason="no_annotations")
                return

    def move_bad_data(
        self, bad_data_dir: str = "bad_data", reason: str = "no_files"
    ) -> None:
        """
        Moves the study's directory to the 'bad_data' directory if the study is incomplete.
        For example no DICOM files or no study text file with annotations.
        """
        print(f"Moving bad data: {self.directory_path} -> Reason: {reason}")

        destination_dir = os.path.join(bad_data_dir, reason)
        os.makedirs(destination_dir, exist_ok=True)

        destination_path = os.path.join(
            destination_dir, os.path.basename(self.directory_path)
        )
        shutil.move(self.directory_path, destination_path)

        self.directory_path = destination_path

    def get_study_uid(self) -> str:
        """Extracts the StudyInstanceUID from the first DICOM file."""
        first_file = self.dicom_files[0]
        if not hasattr(first_file, "StudyInstanceUID"):
            return "Unknown Study UID"
        return first_file.StudyInstanceUID

    def is_dicom_file(self, file_path: str, file_name: str) -> bool:
        return os.path.isfile(file_path) and file_name.lower().endswith("dcm")

    def dicom_file_has_x_attribute(self, x_attr: str, dicom_data: FileDataset) -> bool:
        return (
            hasattr(dicom_data, "SeriesDescription")
            and x_attr in dicom_data.SeriesDescription.upper()
        ) or (
            hasattr(dicom_data, "ProtocolName")
            and x_attr in dicom_data.ProtocolName.upper()
        )

    def is_t1_dicom_file(self, dicom_data: FileDataset) -> bool:
        return self.dicom_file_has_x_attribute("T1", dicom_data)

    def is_t2_dicom_file(self, dicom_data: FileDataset) -> bool:
        return self.dicom_file_has_x_attribute("T2", dicom_data)

    def is_t2_tse_dicom_file(self, dicom_data: FileDataset) -> bool:
        return self.is_t2_dicom_file(dicom_data) and self.dicom_file_has_x_attribute(
            "TSE", dicom_data
        )

    def is_t2_stir_dicom_file(self, dicom_data: FileDataset) -> bool:
        return self.is_t2_dicom_file(dicom_data) and self.dicom_file_has_x_attribute(
            "STIR", dicom_data
        )

    def is_sagittal(self, dicom_data: FileDataset) -> bool:
        return self.dicom_file_has_x_attribute("SAG", dicom_data)

    def get_dicom_files(self, sagittal_only=True) -> List[FileDataset]:
        dicom_files: List[FileDataset] = []
        for file_name in self.directory:
            file_path = os.path.join(self.directory_path, file_name)
            if self.is_dicom_file(file_path, file_name):
                dicom_file = pydicom.dcmread(file_path)
                # # DIXON files have bad annotations, skip them
                # if self.dicom_file_has_x_attribute("DIXON", dicom_file):
                #     continue
                if not self.is_sagittal(dicom_file) and sagittal_only:
                    continue
                dicom_files.append(dicom_file)

        return dicom_files

    def group_dicom_files_by_series(self) -> List[Series]:
        """Groups DICOM files into Series based on SeriesInstanceUID."""
        groups: Dict[str, List[FileDataset]] = {}
        for dicom_file in self.dicom_files:
            series_uid = dicom_file.SeriesInstanceUID
            if series_uid not in groups:
                groups[series_uid] = []
            groups[series_uid].append(dicom_file)

        series_list = []
        for series_uid, files in groups.items():
            first_file = files[0]
            if self.is_t1_dicom_file(first_file):
                series_type = MRIType.T1
            elif self.is_t2_tse_dicom_file(first_file):
                series_type = MRIType.T2_TSE
            elif self.is_t2_stir_dicom_file(first_file):
                series_type = MRIType.T2_STIR
            else:
                series_type = MRIType.UNKNOWN

            series_list.append(Series(self.study_uid, series_uid, files, series_type))

        return series_list

    def get_study_text(self) -> Optional[str]:
        """Finds and reads the study text file with annotations if available."""
        for file_name in self.directory:
            file_path = os.path.join(self.directory_path, file_name)
            if os.path.isfile(file_path) and file_path.lower().endswith(".txt"):
                file = open(file_path, "r")
                text = file.read()
                file.close()
                return text

        # return None if no study text file is found
        return None

    def get_annotated_series(self) -> List[AnnotatedSeries]:
        """Parses the study text file and assigns annotations to the correct series."""
        series: Set[Series] = set()

        series_uid = None

        lines = self.study_text.split("\n")

        for line in lines:
            series_match = re.search(
                r"\(0020, 000e\) Series Instance UID\s+UI:\s+([\d\.]+)", line
            )
            if series_match:
                series_uid = series_match.group(1)

            for serie in self.series:
                if serie.series_uid == series_uid:
                    series.add(serie)
                    break

        # filter out not T2-weighted scans
        series = [
            serie
            for serie in series
            if serie.mri_type in [MRIType.T2_TSE, MRIType.T2_STIR]
        ]

        grouped_tracking_coordinates = self.extract_grouped_tracking_coordinates()

        annotated_series = []

        for serie in series:
            annotations = []
            for (
                series_uid,
                slice_uid,
            ), coordinates in grouped_tracking_coordinates.items():
                if series_uid == serie.series_uid:
                    for frame in serie.frames:
                        if frame.SOPInstanceUID == slice_uid:
                            for x, y in coordinates:
                                annotations.append(
                                    Annotation(frame.InstanceNumber, x, y)
                                )
            annotated_series.append(AnnotatedSeries(serie, annotations))

        return annotated_series

    def extract_grouped_tracking_coordinates(
        self,
    ) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
        """
        Parses the study text to extract and group tracking coordinates.

        This function scans the study text to identify and extract:
        - The Series Instance UID for grouping annotations per series.
        - The Referenced SOP Instance UID to link annotations to specific slices.
        - Tracking UIDs that match predefined codes indicating herniation annotations.
        - Floating-point coordinates representing annotation positions.

        The extracted coordinates are grouped by (Series Instance UID, SOP Instance UID) and
        returned as a dictionary mapping each pair to a list of (x, y) coordinate tuples.

        Returns:
            Dict[Tuple[str, str], List[Tuple[float, float]]]:
            A dictionary where keys are (series UID, slice UID) pairs and values are
            lists of (x, y) annotation coordinates.
        """
        results = defaultdict(list)
        series_uid = None
        ref_sop_uid = None

        lines = self.study_text.split("\n")

        for line in lines:
            series_match = re.search(
                r"\(0020, 000e\) Series Instance UID\s+UI:\s+([\d\.]+)", line
            )
            if series_match:
                series_uid = series_match.group(1)

            sop_match = re.search(
                r"\(0008, 1155\) Referenced SOP Instance UID\s+UI:\s+([\d\.]+)", line
            )
            if sop_match:
                ref_sop_uid = sop_match.group(1)

            tracking_match = re.search(r"\(0062,0020\) Tracking UID:\s+([\w-]+)", line)
            if tracking_match:
                tracking_uid = tracking_match.group(1)

                if tracking_uid in TH_LS_CODES + C_TH_CODES:
                    fl_matches = re.findall(
                        r"FL:\s*\[\s*([\d\.]+),\s*([\d\.]+)\s*\]", line
                    )
                    fl_coords = [(float(x), float(y)) for x, y in fl_matches]

                    if series_uid and ref_sop_uid and fl_coords:
                        results[(series_uid, ref_sop_uid)].append(
                            (fl_coords[0], fl_coords[1])
                        )
        return results

    def export(
        self, nifti_dir: str = "dataset", annotations_file: str = "annotations.json"
    ):
        if len(self.dicom_files) > 0 and self.study_text:
            for series in self.annotated_series:
                if len(series.annotations) > 0:
                    series.export_nifti(nifti_dir)
                    series.export_annotations(annotations_file)


if __name__ == "__main__":

    root_dir = "MR LS chrbtice"

    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:
            study = Study(dirpath)
            study.export()
