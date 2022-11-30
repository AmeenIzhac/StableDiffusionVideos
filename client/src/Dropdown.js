import Cookies from "js-cookie"
import DropdownOptions from "./DropdownOption"
import Img2ImgOptions from "./Img2ImgOptions"
import WalkOptions from "./WalkOptions"

export default function Dropdown({
    frames,
    slideFrameChange,
    isImg2Img,
    setisImg2Img,
    width,
    slideWidthChange,
    height,
    slideHeightChange,
    angle,
    slideAngleChange,
    zoom,
    slideZoomChange,
    fps,
    slideFpsChange,
    xShift,
    slidexShiftChange,
    yShift,
    slideyShiftChange,
    noNoises,
    slideNoNoisesChange,
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
                            <input type="range" min="1" max="120" value={frames} className='slider' id="myRange" onChange={slideFrameChange} />
                        </div>
                        <div className='alignCenter'>
                            <p>Width: <span id="demo">{width}</span></p>
                            <input type="range" min="384" step="64" max="1024" value={width} className='slider' id="myRange" onChange={slideWidthChange} />
                        </div>
                        <div className='alignCenter'>
                            <p>Height: <span id="demo">{height}</span></p>
                            <input type="range" min="384" step="64" max="1024" value={height} className='slider' id="myRange" onChange={slideHeightChange} />
                        </div>
                        <div className='slideContainer alignCenter'>
                            <p>Frames per second: <span id="demo">{fps}</span></p>
                            <input type="range" min="1" max="30" value={fps} className='slider' id="myRange" onChange={slideFpsChange} />
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
                                slideAngleChange={slideAngleChange}
                                zoom={zoom}
                                slideZoomChange={slideZoomChange}
                                xShift={xShift}
                                slidexShiftChange={slidexShiftChange}
                                yShift={yShift}
                                slideyShiftChange={slideyShiftChange}
                            /> 
                            : 
                            <></>
                        }

                        { !isImg2Img ?
                            <WalkOptions 
                                noNoises={noNoises}
                                slideNoNoisesChange={slideNoNoisesChange}
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