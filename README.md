# pupillometry
Task: given video with person's face in infrared (or possibly also normal) lightning conditions - find the size of pupil and iris for each eye.  

To run video processing fill config.yaml file and run `python -m src.video_processor --config=config.yaml`

To run processing from laptop webcam: `python -m src.livestream_processor --config=config.yaml`

------
Create conda environment `conda env create -f requirements.yaml` (miniconda or anaconda should be installed)  

Create virtualenv environment:  
`virtualenv pupil_env -p python3`  
`source pupil_env/bin/activate`  
`pip install -r requirements.txt` 