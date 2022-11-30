import Cookies from "js-cookie"
import DropdownOptions from "./DropdownOption"
import Img2ImgOptions from "./Img2ImgOptions"
import WalkOptions from "./WalkOptions"

export default function Dropdown({
    frames,
    setFrames,
    isImg2Img,
    setisImg2Img,
    width,
    setWidth,
    height,
    setHeight,
    angle,
    setAngle,
    zoom,
    setZoom,
    fps,
    setFps,
    xShift,
    setxShift,
    yShift,
    setyShift,
    noNoises,
    setNoNoises,
    slideStateChange,
    dropOptions
}) {
    
    return (
        <div className='slideOptions'>
            {Cookies.get("loggedInUser") ?
                <>
                    <div className='dropdownOption' id="dropdown">
                        <DropdownOptions 
                            isImg2Img={isImg2Img}
                            setisImg2Img={setisImg2Img}
                        />
                        <div className='slideContainer alignCenter'>
                            <p>Number of Frames: <span id="demo">{frames}</span></p>
                            <input type="range" min="1" max="120" value={frames} className='slider' id="myRange" onChange={e => slideStateChange(e, setFrames)} />
                        </div>
                        <div className='alignCenter'>
                            <p>Width: <span id="demo">{width}</span></p>
                            <input type="range" min="384" step="64" max="1024" value={width} className='slider' id="myRange" onChange={e => slideStateChange(e, setWidth)} />
                        </div>
                        <div className='alignCenter'>
                            <p>Height: <span id="demo">{height}</span></p>
                            <input type="range" min="384" step="64" max="1024" value={height} className='slider' id="myRange" onChange={e => slideStateChange(e, setHeight)} />
                        </div>
                        <div className='slideContainer alignCenter'>
                            <p>Frames per second: <span id="demo">{fps}</span></p>
                            <input type="range" min="1" max="30" value={fps} className='slider' id="myRange" onChange={e => slideStateChange(e, setFps)} />
                        </div>
                        <div className="alignCenter">
                            <div className="checkBoxContainer">
                                <label htmlFor="upscale">Upscales? </label>
                                <input type="checkbox" id="upscale" name="upscale" />
                            </div>
                        </div>
                        { isImg2Img ? 
                            <Img2ImgOptions 
                                angle={angle} 
                                setAngle={setAngle}
                                zoom={zoom}
                                setZoom={setZoom}
                                xShift={xShift}
                                setxShift={setxShift}
                                yShift={yShift}
                                setyShift={setyShift}
                                slideStateChange={slideStateChange}
                            /> 
                            : 
                            <></>
                        }

                        { !isImg2Img ?
                            <WalkOptions 
                                noNoises={noNoises}
                                setNoNoises={setNoNoises}
                                slideStateChange={slideStateChange}
                            />
                            :
                            <></>
                        }
                        
                        
                    </div>
                    <button className='dropArrow' onClick={dropOptions} id="button"></button>
                </>
                :
                <></>
            }
        </div>
    )
}