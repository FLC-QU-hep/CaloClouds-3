import sys
from pointcloud.utils import stats_accumulator

assert len(sys.argv) == 3, "Please provide the number of sections and the section number"
total_sections, section_number = sys.argv[1:]
total_sections = int(total_sections)
section_number = int(section_number)
stats_accumulator.read_section(total_sections, section_number)
