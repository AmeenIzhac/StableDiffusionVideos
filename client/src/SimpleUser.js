import Navbar from "./Navbar";
import { useState } from "react";
import axios from "axios";
import loadingAnimation from "./assets/loading.gif";
import Cookies from 'js-cookie'
import Dropdown from "./Dropdown";
import * as qs from 'qs';


export default function SimpleUser() {
  const [src, setSrc] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [frames, setFrames] = useState("30")
  const [width, setWidth] = useState("704")
  const [height, setHeight] = useState("704")
  const [angle, setAngle] = useState("0")
  const [zoom, setZoom] = useState("1")
  const [loggedIn, setLoggedIn] = useState(Cookies.get("loggedInUser") != null)
  const [prompts, setPrompts] = useState([])


  const handleChange = (e) => {
    setPrompt(e.target.value);
  };

  async function getVideo() {
    if (prompt === "") {
      alert("Please enter a prompt");
      return;
    }

    setSrc("");
    setLoading(true);
    axios({
      method: "get",
      // url: `https://stablediffusionvideoswebserver-production.up.railway.app/generate`,
      url: `http://localhost:3001/generate`,
      params: {
        prompts: [...prompts, prompt],
        frames: frames,
        width: width,
        height: height,
        angle: angle,
        zoom: zoom
      },
      responseType: "blob",
      timeout: 10000000
    })
      .then((response) => {
        console.log(response.data);
        setSrc(URL.createObjectURL(response.data));
        console.log(src);
      })
      .catch((error) => {
        alert("Could not connect to server");
      })
      .finally(() => {
        setLoading(false);
      });

    setAngle("0")
    setZoom("1")
    setWidth("704")
    setHeight("704")
    setPrompt("")
    setPrompts([])
    const dropdown = document.getElementById("dropdown")
    const button = document.getElementById("button")
    dropdown.classList.remove("open")
    button.classList.remove("rotate")
  }

  function addPrompt() {
    if (prompt === "") {
      alert("Please enter a prompt");
      return;
    }
    setPrompts([...prompts, prompt])
    setPrompt("")
  }

  //dropdown code
  const dropOptions = () => {
    const dropdown = document.getElementById("dropdown")
    const button = document.getElementById("button")
    const flex = document.getElementById("flex")
    if (!dropdown.classList.contains("open")) {
      flex.style.flexDirection = 'row'
      flex.style.alignItems = 'initial'
    } else {
      setTimeout(() => {
        flex.style.flexDirection = 'column'
        flex.style.alignItems = 'center'
      }, 500);
    }
    dropdown.classList.toggle("open")
    button.classList.toggle("rotate")
  }

  const slideFrameChange = (e) => {
    setFrames(e.target.value);
  }

  const slideWidthChange = (e) => {
    setWidth(e.target.value)
  }
  const slideHeightChange = (e) => {
    setHeight(e.target.value)
  }

  const slideAngleChange = (e) => {
    setAngle(e.target.value)
  }

  const slideZoomChange = (e) => {
    setZoom(e.target.value)
  }

  const logger = (e) => {
    e.preventDefault()
    const numberRegex = new RegExp('[0-9]+$')
    console.log(`Prompt is ${prompt}`)
    console.log(`Frames are ${frames}`);
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
            <button className='promptButton' onClick={addPrompt}>+</button>
            <button className='promptButton' onClick={getVideo}>Generate Video</button>
            {/* <button className='promptButton' onClick={getVideo} onSubmit={logger}>Generate Video</button> */}
          </div>
          <div className="promptsContainer">
            {prompts.map((prompt, index) => {
              return <div key={index} className="promptsList">
                <span> {prompt} </span>
                <button
                  onClick={() => { setPrompts(prompts.filter((_, i) => i !== index)) }}
                  className="removePrompt">
                  X
                </button>
              </div>
            })}
          </div>
          <div className="flex" id="flex">
            <Dropdown
              frames={frames}
              slideFrameChange={slideFrameChange}
              width={width}
              slideWidthChange={slideWidthChange}
              height={height}
              slideHeightChange={slideHeightChange}
              zoom={zoom}
              slideZoomChange={slideZoomChange}
              angle={angle}
              slideAngleChange={slideAngleChange}
              dropOptions={dropOptions}
            />
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
      </div>
    </div>
  );
}

