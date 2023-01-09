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
    fMult,
    setfMult,
    strength,
    setStrength,
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
                            <p className="tooltip">Number of Frames : <span id="demo">{frames}</span>
                                <span class="tooltiptext">The number of frames generated before motion interpolation</span>
                            </p>
                            <input type="range" min="1" max="120" value={frames} className='slider' id="myRange" onChange={e => slideStateChange(e, setFrames)} />
                        </div>
                        <div className='alignCenter'>
                            <p className="tooltip">Width : <span id="demo">{width}</span>
                                <span class="tooltiptext">The width of the generated video before upscaling</span>
                            </p>
                            <input type="range" min="384" step="64" max="1024" value={width} className='slider' id="myRange" onChange={e => slideStateChange(e, setWidth)} />
                        </div>
                        <div className='alignCenter'>
                            <p className="tooltip">Height : <span id="demo">{height}</span>
                                <span class="tooltiptext">The height of the generated video before upscaling</span>
                            </p>
                            <input type="range" min="384" step="64" max="1024" value={height} className='slider' id="myRange" onChange={e => slideStateChange(e, setHeight)} />
                        </div>
                        <div className='slideContainer alignCenter'>
                            <p className="tooltip">Frames per second : <span id="demo">{fps}</span>
                                <span class="tooltiptext">The framerate of the final video</span>
                            </p>
                            <input type="range" min="1" max="30" value={fps} className='slider' id="myRange" onChange={e => slideStateChange(e, setFps)} />
                        </div>
                        <div className='slideContainer alignCenter'>
                            <p className="tooltip">Frames Multiplier : <span id="demo">{Math.pow(2,fMult)}</span>
                                <span class="tooltiptext">The multiplier used to perform motion interpolation (adding frames between the generated ones). For example, if Number of Frames is 30 and this setting is 4, the final video will have 120 images.</span>
                            </p>
                            <input type="range" min="0" max="4" value={fMult} className='slider' id="myRange" onChange={e => slideStateChange(e, setfMult)} />
                        </div>
                        <div className="alignCenter">
                          <div className="tooltip checkBoxContainer">
                            <label htmlFor="upscale">Upscales? </label>
                            <input type="checkbox" id="upscale" name="upscale" />
                            <span class="tooltiptext">Whether to upscale the generated images. The image definition is doubled</span>
                          </div>
                        </div>

                        {isImg2Img ?
                            <Img2ImgOptions
                                angle={angle}
                                setAngle={setAngle}
                                zoom={zoom}
                                setZoom={setZoom}
                                xShift={xShift}
                                setxShift={setxShift}
                                yShift={yShift}
                                setyShift={setyShift}
                                strength={strength}
                                setStrength={setStrength}
                                slideStateChange={slideStateChange}
                            />
                            :
                            <WalkOptions
                                noNoises={noNoises}
                                setNoNoises={setNoNoises}
                                slideStateChange={slideStateChange}
                            />
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