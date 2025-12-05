"""
Solar Panel Detection System 

                       _________        _______           ________          _____ ______                                                 
                      │╲___   ___╲     │╲  ___ ╲         │╲   __  ╲        │╲   _ ╲  _   ╲                                               
                      ╲│___ ╲  ╲_│     ╲ ╲   __╱│        ╲ ╲  ╲│╲  ╲       ╲ ╲  ╲╲╲__╲ ╲  ╲                                              
                           ╲ ╲  ╲       ╲ ╲  ╲_│╱__       ╲ ╲   __  ╲       ╲ ╲  ╲╲│__│ ╲  ╲                                             
                            ╲ ╲  ╲       ╲ ╲  ╲_│╲ ╲       ╲ ╲  ╲ ╲  ╲       ╲ ╲  ╲    ╲ ╲  ╲                                            
                             ╲ ╲__╲       ╲ ╲_______╲       ╲ ╲__╲ ╲__╲       ╲ ╲__╲    ╲ ╲__╲                                           
                              ╲│__│        ╲│_______│        ╲│__│╲│__│        ╲│__│     ╲│__│                        
                                                                                                                      
 ________          ___  __            ________          _________        ________           ___  ___          ___  __            ___     
│╲   __  ╲        │╲  ╲│╲  ╲         │╲   __  ╲        │╲___   ___╲     │╲   ____╲         │╲  ╲│╲  ╲        │╲  ╲│╲  ╲         │╲  ╲    
╲ ╲  ╲│╲  ╲       ╲ ╲  ╲╱  ╱│_       ╲ ╲  ╲│╲  ╲       ╲│___ ╲  ╲_│     ╲ ╲  ╲___│_        ╲ ╲  ╲╲╲  ╲       ╲ ╲  ╲╱  ╱│_       ╲ ╲  ╲   
 ╲ ╲   __  ╲       ╲ ╲   ___  ╲       ╲ ╲   __  ╲           ╲ ╲  ╲       ╲ ╲_____  ╲        ╲ ╲  ╲╲╲  ╲       ╲ ╲   ___  ╲       ╲ ╲  ╲  
  ╲ ╲  ╲ ╲  ╲       ╲ ╲  ╲╲ ╲  ╲       ╲ ╲  ╲ ╲  ╲           ╲ ╲  ╲       ╲│____│╲  ╲        ╲ ╲  ╲╲╲  ╲       ╲ ╲  ╲╲ ╲  ╲       ╲ ╲  ╲ 
   ╲ ╲__╲ ╲__╲       ╲ ╲__╲╲ ╲__╲       ╲ ╲__╲ ╲__╲           ╲ ╲__╲        ____╲_╲  ╲        ╲ ╲_______╲       ╲ ╲__╲╲ ╲__╲       ╲ ╲__╲
    ╲│__│╲│__│        ╲│__│ ╲│__│        ╲│__│╲│__│            ╲│__│       │╲_________╲        ╲│_______│        ╲│__│ ╲│__│        ╲│__│
                                                                           ╲│_________│                                                  
                                                                                                                                         
computer vision pipeline for automated rooftop 
solar panel detection using satellite imagery and deep learning.
"""

import logging
from src.pipeline import SolarDetectionPipeline


def setup_logging():
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point for the detection pipeline."""
    setup_logging()
    print("""
               _____   _____      _      __  __                
              |_   _| | ____|    / \    |  \/  |               
                | |   |  _|     / _ \   | |\/| |               
                | |   | |___   / ___ \  | |  | |               
                |_|   |_____| /_/   \_\ |_|  |_|               
                                                               
    _      _  __     _      _____   ____    _   _   _  __  ___ 
   / \    | |/ /    / \    |_   _| / ___|  | | | | | |/ / |_ _|
  / _ \   | ' /    / _ \     | |   \___ \  | | | | | ' /   | | 
 / ___ \  | . \   / ___ \    | |    ___) | | |_| | | . \   | | 
/_/   \_\ |_|\_\ /_/   \_\   |_|   |____/   \___/  |_|\_\ |___|

""")
    
    
    try:
        pipeline = SolarDetectionPipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()