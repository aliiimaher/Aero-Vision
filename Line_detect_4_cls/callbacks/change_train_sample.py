class SampleInfoCallback:
    def __init__(self):
        # Initialize with an empty dictionary to store sample information
        self.sample_info = {}
    def modify_sample_info(self):
        """
        Allow the user to adjust the parameters.
        """
        # Loop through each key in the sample_info dictionary
        for key in self.sample_info.keys():
            # Prompt the user for a new value or to keep the current value
            new_value = input(f"Current {key}: {self.sample_info[key]} - Enter new value (or just press Enter to keep current): ")
            if new_value:
                # Convert new value to the appropriate type based on the key
                if key in ["rotation_angle", "scale_val"]:
                    self.sample_info[key] = float(new_value)  # Convert to float for these keys
                elif key == "flipy":
                    self.sample_info[key] = new_value.lower() in ['true', '1', 'yes']  # Convert to boolean
                elif key == "sample_center":
                    self.sample_info[key] = tuple(map(float, new_value.split(',')))  # Convert to tuple of floats
                elif key == "img_file":
                    self.sample_info[key] = new_value  # Keep as string for file path

        # Print the updated parameters
        print("New parameters have been set:")
        for key, value in self.sample_info.items():
            print(f"{key}: {value}")
