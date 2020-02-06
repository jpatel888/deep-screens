def get_data_summary():
    return """
        File Structure
            data/
                (yyyy.MM.dd.hh.mm.ss)_(baseline).jpg
                (yyyy.MM.dd.hh.mm.ss)_(current).jpg
                (yyyy.MM.dd.hh.mm.ss)_(labels).jpg
        
        Images
            nd_array len(shape) == 3
        
        Label JSON:
            {
                "image_width": [screenshot_width],
                "image_height": [screenshot_height],
                "activity_name": [activity_name],
                "scroll_position": [scroll_position (if applicable)],
                "defects": [
                    {
                        "defect_type": [defect_type],
                        "location": [midpoint_x, midpoint_y, width, height]
                    },
                    {
                        "defect_type": [defect_type],
                        "location": [midpoint_x, midpoint_y, width, height]
                    }
                ]
            }
    """
