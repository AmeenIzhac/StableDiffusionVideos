import Cookies from "js-cookie"

export default function Dropdown({
    frames,
    slideFrameChange,
    isImg2Img,
    setisImg2Img,
    isLatentWalk,
    setisLatentWalk,
    isSemanticWalk,
    setisSemanticWalk,
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
    dropOptions
}) {
    const dark = '#181818'
    const light = '#9A4EAE'
    const defaultStyle = {
        backgroundColor: light,
        color: dark
    }
    const chosenStyle = {
        backgroundColor: dark,
        color:light,
        borderTop: '1px solid ' + light
    }
    const img2imgStyle = isImg2Img ? chosenStyle : defaultStyle 
    const latentStyle = isLatentWalk ? chosenStyle : defaultStyle 
    const semanticStyle = isSemanticWalk ? chosenStyle : defaultStyle 

    const chooseImg2Img = () => {
        setisImg2Img(true)
        setisLatentWalk(false)
        setisSemanticWalk(false)
    }
    const chooseLatent = () => {
        setisImg2Img(false)
        setisLatentWalk(true)
        setisSemanticWalk(false)
    }
    const chooseSemantic = () => {
        setisImg2Img(false)
        setisLatentWalk(false)
        setisSemanticWalk(true)
    }
    return (
        <div className='slideOptions'>
            {Cookies.get("loggedInUser") ?
                <>
                    <div className='dropdownOption' id="dropdown">
                        <div className="dropdownVidOption">
                            <ul className="vidOptionContainer">
                                <li className="img2img">
                                    <button className="vidOptionBtn" style={img2imgStyle} onClick={chooseImg2Img}>img2img</button>
                                </li>
                                <li className="latentWalk">
                                    <button className="vidOptionBtn" style={latentStyle} onClick={chooseLatent}>Latent Walk</button>
                                </li>
                                <li className="semanticWalk">
                                    <button className="vidOptionBtn" style={semanticStyle} onClick={chooseSemantic}>Semantic Walk</button>
                                </li>
                            </ul>
                        </div>
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
                        <div className='alignCenter'>
                            <p>Angle: <span id="demo">{angle}</span></p>
                            <input type="range" min="-10" step="1" max="10" value={angle} className='slider' id="myRange" onChange={slideAngleChange} />
                        </div>
                        <div className='alignCenter'>
                            <p>Zoom: <span id="demo">{zoom}</span></p>
                            <input type="range" min="0.7" step="0.05" max="1.3" value={zoom} className='slider' id="myRange" onChange={slideZoomChange} />
                        </div>
                        <div className='slideContainer alignCenter'>
                            <p>Frames per second: <span id="demo">{fps}</span></p>
                            <input type="range" min="1" max="30" value={fps} className='slider' id="myRange" onChange={slideFpsChange} />
                        </div>
                        <div className='alignCenter'>
                            <p>x-shift: <span id="demo">{xShift}</span></p>
                            <input type="range" min="-10" step="1" max="10" value={xShift} className='slider' id="myRange" onChange={slidexShiftChange} />
                        </div>
                        <div className='alignCenter'>
                            <p>y-shift: <span id="demo">{yShift}</span></p>
                            <input type="range" min="-10" step="1" max="10" value={yShift} className='slider' id="myRange" onChange={slideyShiftChange} />
                        </div>
                        <div className="alignCenter">
                            <input type="checkbox" id="scales" name="scales"  />
                            <label for="scales">Scales</label>
                        </div>
                    </div>
                    <button className='dropArrow' onClick={dropOptions} id="button"></button>
                </>
                :
                <></>
            }
        </div>
    )
}