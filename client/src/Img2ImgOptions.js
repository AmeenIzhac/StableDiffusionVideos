

export default function Img2ImgOptions({
  angle,
  setAngle,
  zoom,
  setZoom,
  xShift,
  setxShift,
  yShift,
  setyShift,
  strength,
  setStrength,
  slideStateChange
}) {
  return (
    <div className="extraOptions">
      <h3 className="fullRow optionHeader">Img2Img Specific Options</h3>
      <div className='alignCenter'>
          <p className="tooltip">Angle : <span id="demo">{angle}</span>
            <span class="tooltiptext">The angle of rotation between 2 frames in degrees</span>
            </p>
          <input type="range" min="-20" step="0.1" max="20" value={angle} className='slider' id="myRange" onChange={e => slideStateChange(e, setAngle)} />
      </div>
      <div className='alignCenter'>
          <p className="tooltip">Zoom : <span id="demo">{zoom}</span>
            <span class="tooltiptext">The zoom factor between 2 frames. Smaller than 1 is zoom out, larger is zoom in, 1 is no zoom</span>
            </p>
          <input type="range" min="0.5" step="0.001" max="1.5" value={zoom} className='slider' id="myRange" onChange={e => slideStateChange(e, setZoom)} />
      </div>
      <div className='alignCenter'>
          <p className="tooltip">Horizontal shift : <span id="demo">{xShift}</span>
              <span class="tooltiptext">The horizontal translation (in pixels) between 2 frames</span>
              </p>
          <input type="range" min="-50" step="0.1" max="50" value={xShift} className='slider' id="myRange" onChange={e => slideStateChange(e, setxShift)} />
      </div>
      <div className='alignCenter'>
          <p className="tooltip">Vertical shift : <span id="demo">{yShift}</span>
            <span class="tooltiptext">The vertical translation (in pixels) between 2 frames</span>
            </p>
          <input type="range" min="-50" step="0.1" max="50" value={yShift} className='slider' id="myRange" onChange={e => slideStateChange(e, setyShift)} />
      </div>
      <div className='alignCenter'>
          <p className="tooltip">Strength (advanced) : <span id="demo">{strength}</span>
            <span class="tooltiptext">The strength at which each image is modified from the previous one. 0 means a static image, 1 means a completely different frame each time. Reasonable values in 0.3 - 0.5</span>
            </p>
          <input type="range" min="0" step="0.01" max="1" value={strength} className='slider' id="myRange" onChange={e => slideStateChange(e, setStrength)} />
      </div>
    </div>
    
  )
}