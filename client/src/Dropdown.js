import Cookies from "js-cookie"

export default function Dropdown({
    frames,
    slideFrameChange,
    width,
    slideWidthChange,
    height,
    slideHeightChange,
    angle,
    slideAngleChange,
    zoom,
    slideZoomChange,
    dropOptions
}) {
    return (
        <div className='slideOptions'>
            {Cookies.get("loggedInUser") ?
                <>
                    <div className='dropdownOption' id="dropdown">
                        <form>
                            <div className='slideContainer alignCenter'>
                                <p>Number of Frames: <span id="demo">{frames}</span></p>
                                <input type="range" min="1" max="60" value={frames} className='slider' id="myRange" onChange={slideFrameChange} />
                            </div>
                            <hr />
                            <div className='alignCenter'>
                                <p>Width: <span id="demo">{width}</span></p>
                                <input type="range" min="384" step="64" max="1024" value={width} className='slider' id="myRange" onChange={slideWidthChange} />
                            </div>
                            <hr />
                            <div className='alignCenter'>
                                <p>Height: <span id="demo">{height}</span></p>
                                <input type="range" min="384" step="64" max="1024" value={height} className='slider' id="myRange" onChange={slideHeightChange} />
                            </div>
                            <hr />
                            <div className='alignCenter'>
                                <p>Angle: <span id="demo">{angle}</span></p>
                                <input type="range" min="-10" step="1" max="10" value={angle} className='slider' id="myRange" onChange={slideAngleChange} />
                            </div>
                            <hr />
                            <div className='alignCenter'>
                                <p>Zoom: <span id="demo">{zoom}</span></p>
                                <input type="range" min="0.7" step="0.05" max="1.3" value={zoom} className='slider' id="myRange" onChange={slideZoomChange} />
                            </div>
                        </form>
                    </div>
                    <button className='dropArrow' onClick={dropOptions} id="button"></button>
                </>
                :
                <></>
            }
        </div>
    )
}