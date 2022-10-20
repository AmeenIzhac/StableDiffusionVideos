import Navbar from './Navbar'
import './SuperUser.css'

export default function SuperUser() {
    return (
        <>
            <Navbar isSuper={true} link="Simple User" href="/" />
            <div className="mainDiv">
                <div  className='promptDiv'>
                    <input className='prompt' placeholder='Enter Text Prompt...'></input>
                </div>
                <div className='output_container'>
                    <div className='controls_container'>
                        <h1>testing this thing</h1>
                    </div>
                    <div className='videoDiv'>
                        <video className="video" controls>
                            <source src="vid.mp4" type="video/mp4"/>
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    
                </div>
                
            </div>
        </>
    );
  }

  