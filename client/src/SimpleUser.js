import Navbar from './Navbar'

export default function SimpleUser() {
    return (
        <div className='SimpleUser'>
            <Navbar isSuper={false} link="Super User" href="superUser"/>
            <div className="mainDiv">
                <div  className='promptDiv'>
                    <input className='prompt' placeholder='Enter Text Prompt...'></input>
                </div>
                <div className='videoDiv'>
                    <video className="video" controls>
                        <source src="vid.mp4" type="video/mp4"/>
                        Your browser does not support the video tag.
                    </video>
                </div>
            </div>
        </div>
    );
  }

  