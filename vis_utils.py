import numpy as np
import plotly.express as px

def visualise_attended_tokens(attended_tokens: np.ndarray):
    section_length = 64
    section_count = 256 // section_length
    for section_ind in range(section_count):
        arr = attended_tokens[:, section_ind * section_length: (section_ind + 1) * section_length]
        fig = px.imshow(arr, text_auto=True)
        fig.show()