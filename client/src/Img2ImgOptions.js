

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
          <p>Angle: <span id="demo">{angle}</span></p>
          <input type="range" min="-10" step="0.1" max="10" value={angle} className='slider' id="myRange" onChange={e => slideStateChange(e, setAngle)} />
      </div>
      <div className='alignCenter'>
          <p>Zoom: <span id="demo">{zoom}</span></p>
          <input type="range" min="0.7" step="0.001" max="1.3" value={zoom} className='slider' id="myRange" onChange={e => slideStateChange(e, setZoom)} />
      </div>
      <div className='alignCenter'>
          <p>x-shift: <span id="demo">{xShift}</span></p>
          <input type="range" min="-10" step="0.1" max="10" value={xShift} className='slider' id="myRange" onChange={e => slideStateChange(e, setxShift)} />
      </div>
      <div className='alignCenter'>
          <p>y-shift: <span id="demo">{yShift}</span></p>
          <input type="range" min="-10" step="0.1" max="10" value={yShift} className='slider' id="myRange" onChange={e => slideStateChange(e, setyShift)} />
      </div>
      <div className='alignCenter'>
          <p>Strength (advanced): <span id="demo">{strength}</span></p>
          <input type="range" min="0" step="0.01" max="1" value={strength} className='slider' id="myRange" onChange={e => slideStateChange(e, setStrength)} />
      </div>
    </div>
    
  )
}