import Navbar from "./Navbar";
import { useState } from "react";
import axios from "axios";
import loadingAnimation from "./assets/painting.gif";
import Cookies from 'js-cookie'
import FrameSelect from "./components/FrameSelect";
import a from "./assets/a.jpg";
import b from "./assets/b.jpg";
import c from "./assets/c.jpg";
import d from "./assets/d.jpg";

export default function SimpleUser() {
  const [src, setSrc] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [frameNumber, setFrameNumber] = useState("30")
  const [width, setWidth] = useState("")
  const [height, setHeight] = useState("")
  const [loggedIn, setLoggedIn] = useState(Cookies.get("loggedInUser") != null)


  const handleChange = (e) => {
    setPrompt(e.target.value);
  };

  // temp for testing
  const frames = [a, b, c, d];
  const selectFunction = (index) => {
    console.log(`Frame ${index} Selected`);
  }
  const getNewFrame = () => {
    console.log("Get new frame");
  }
  // end temp testing

  function getVideo() {
    if (prompt === "") {
      alert("Please enter a prompt");
      return;
    }

    setLoading(true);
    axios({
      method: "get",
      url: `https://stablediffusionvideoswebserver-production.up.railway.app/generate?prompt=${prompt}`,
      responseType: "blob",
      timeout: 10000000,
    })
      .then((response) => {
        console.log(response.data);
        setSrc(URL.createObjectURL(response.data));
        console.log(src);
      })
      .catch((error) => {
        console.log(error);
      })
      .finally(() => {
        setLoading(false);
      });


    setPrompt("");
  }

  //dropdown code
  const dropOptions = () => {
    const dropdown = document.getElementById("dropdown")
    const button = document.getElementById("button")
    dropdown.classList.toggle("open")
    button.classList.toggle("rotate")
  }

  const slideChange = (e) => {
    setFrameNumber(e.target.value);
  }

  const widthChange = (e) => {
    setWidth(e.target.value)
  }
  const heightChange = (e) => {
    setHeight(e.target.value)
  }

  const logger = (e) => {
    e.preventDefault()
    const numberRegex = new RegExp('[0-9]+$')
    console.log(`Prompt is ${prompt}`)
    console.log(`Frames are ${frameNumber}`);
    console.log(`Width is ${width}`)
    if (!(numberRegex.test(width) && numberRegex.test(height))) {
      alert('Please Enter Width and Height as Integer Values')
      return;
    } else {
      const h = parseInt(height)
      const w = parseInt(width)
      if (h % 64 !== 0 || w % 64 !== 0) {
        alert('Width and Height should be multiples of 64')
        return;
      }
    }
    console.log(`Width is ${height}`)
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter') {
      logger(event)
    }
  }



  return (
    <div className='SimpleUser'>
      <Navbar loggedIn={loggedIn} setLoggedIn={setLoggedIn} />

      <div className="mainDiv">
        <div className='promptContainerDiv'>
          <div className='promptDiv'>
            <input className='prompt' value={prompt} placeholder='Enter Text Prompt...' onChange={handleChange} onSubmit={getVideo} onKeyDown={handleKeyDown}></input>
            <button className='promptButton' onClick={getVideo}>Generate Video</button>
            {/* <button className='promptButton' onClick={getVideo} onSubmit={logger}>Generate Video</button> */}
          </div>
          <div className='slideOptions'>
            {Cookies.get("loggedInUser") ?
              <>
                <div className='dropdownOption' id="dropdown">
                  <form>
                    <div className='slideContainer alignCenter'>
                      <p>Number of Frames:</p>
                      <input type="range" min="1" max="60" value={frameNumber} className='slider' id="myRange" onChange={slideChange} />
                      <p>Value: <span id="demo">{frameNumber}</span></p>
                    </div>
                    <hr />
                    <div className='alignCenter'>
                      <input className='dropdownInput alignCenter' value={width} placeholder='Enter Width' onChange={widthChange} />
                    </div>
                    <hr />
                    <div className='alignCenter'>
                      <input className='dropdownInput alignCenter' value={height} placeholder='Enter Height' onChange={heightChange} />
                    </div>
                  </form>
                </div>
                <div>
                  <FrameSelect srcs={frames} selectFunction={selectFunction} getNewFrame={getNewFrame}></FrameSelect>
                </div>
                <div className='videoDiv'>
                  {src ?
                    <video id="vidObj" width="500" height="360" controls loop muted autoPlay>
                      <source src={src} type="video/mp4" />
                    </video>
                    :
                    (loading ?
                      <img src={loadingAnimation} alt='loading thingy' /> : null)
                  }
                </div>
                <button className='dropArrow' onClick={dropOptions} id="button"></button>
              </>
              :
              <></>
            }

          </div>
        </div>
        <div className='videoDiv'>
          {src ?
            <video id="vidObj" width="500" height="360" controls loop muted autoPlay>
              <source src={src} type="video/mp4" />
            </video>
            :
            (loading ?
              <img src={loadingAnimation} alt='loading thingy' /> : null)
          }

        </div>
      </div>
    </div>
  );
}

