import json

class ConfigParser():
    def __init__(self, config_path:str = './config.json') -> None:
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Config file not found")
        
#=============
# RGB INPUT CONFIG

    def get_rgb_input_tensor_shape(self) -> tuple:
        rgb_input_config = self.config['rgb_input_config']
        return (rgb_input_config['chanels'], rgb_input_config['cols'], rgb_input_config['rows'])
    
    def get_rgb_input_cv2_shape(self) -> tuple:
        rgb_input_config = self.config['rgb_input_config']
        return (rgb_input_config['cols'], rgb_input_config['rows'], rgb_input_config['chanels'])
    
    def get_rgb_input_gsd_m(self) -> float:
        rgb_input_config = self.config['rgb_input_config']
        return rgb_input_config['gsd_m']
    
#=============
# MULTISPECTRAL INPUT CONFIG
    
    def get_mul_input_tensor_shape(self) -> tuple:
        mult_input_config = self.config['multispectral_input_config']
        return (mult_input_config['chanels'], mult_input_config['cols'], mult_input_config['rows'])
    
    def get_mul_input_cv2_shape(self) -> tuple:
        mult_input_config = self.config['multispectral_input_config']
        return (mult_input_config['cols'], mult_input_config['rows'], mult_input_config['chanels'])
    
    def get_mul_input_gsd_m(self) -> float:
        mult_input_config = self.config['multispectral_input_config']
        return mult_input_config['gsd_m']
    
#=============
# OUTPUT CONFIG

    def get_output_tensor_shape(self) -> tuple:
        output_config = self.config['output_config']
        return (
            output_config['chanels'],
            output_config['cols'],
            output_config['rows'],
        )

    def get_output_cv2_shape(self) -> tuple:
        output_config = self.config['output_config']
        return (
            output_config['cols'],
            output_config['rows'],
            output_config['chanels'],
        )

    def get_output_gsd_m(self) -> float:
        output_config = self.config['output_config']
        return output_config['gsd_m']
    
#=============
# SENTINEL 2 CONFIG

    def get_sentinel_2_bands(self):
        sentinel_2_config = self.config['sentinel2_l2a_config']
        return sentinel_2_config['bands_list']

#=============
# TRAINING CONFIG

    def get_training_num_workers(self) -> int:
        training_config = self.config['traininig']
        return training_config['num_workers']

#=============
# CHUNKINMG CONFIG

    def get_chunk_size(self):
        chunking_config = self.config['chunking_config']
        return chunking_config['chunk_shape']


#====================================================================================================
def test_class_init():
    config_path = 'config.json'
    config = ConfigParser(config_path)

#=============
# RGB INPUT CONFIG

def test_get_rgb_input_cv2_shape():
    config_path = 'config.json'
    config = ConfigParser(config_path)
    shape = config.get_rgb_input_cv2_shape()
    assert shape[2] == 3
    assert shape[0] > 32
    assert shape[1] > 32

def test_get_rgb_input_tensor_shape():
    config_path = 'config.json'
    config = ConfigParser(config_path)
    shape = config.get_rgb_input_tensor_shape()
    assert shape[0] == 3
    assert shape[1] > 32
    assert shape[2] > 32

def test_rgb_gsd():
    config_path = 'config.json'
    config = ConfigParser(config_path)
    gsd = config.get_rgb_input_gsd_m()
    assert gsd > 0 and gsd < 100

#=============
# MULTISPECTRAL INPUT CONFIG

def test_get_mul_input_cv2_shape():
    config_path = 'config.json'
    config = ConfigParser(config_path)
    shape = config.get_mul_input_cv2_shape()
    assert shape[2] == 4
    assert shape[0] > 32 
    assert shape[1] > 32  

def test_get_mul_input_tensor_shape():
    config_path = 'config.json'
    config = ConfigParser(config_path)
    shape = config.get_mul_input_tensor_shape()
    assert shape[0] == 4
    assert shape[1] > 32
    assert shape[2] > 32

def test_mul_gsd():
    config_path = 'config.json'
    config = ConfigParser(config_path)
    gsd = config.get_mul_input_gsd_m()
    assert gsd > 0 and gsd < 100
    
#=============
# SENTINEL 2 CONFIG

def test_get_sentinel_2_bands():
    config_path = 'config.json'
    config = ConfigParser(config_path)
    bands = config.get_sentinel_2_bands()
    assert len(bands) == 5