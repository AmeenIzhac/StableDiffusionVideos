
export default function WalkOptions({
  noNoises,
  setNoNoises,
  slideStateChange
}) {
  return (
    <div className="extraOptions">
      <h3 className="fullRow optionHeader">Walk Specific Options</h3>
      <div className='alignCenter'>
          <p className="tooltip">Latent Walking Speed: 
            <span id="demo">{noNoises}</span>
            <span class="tooltiptext">Tooltip text</span>
          </p>
          <input type="range" min="1" step="1" max="10" value={noNoises} className='slider' id="myRange" onChange={e => slideStateChange(e, setNoNoises)} />
      </div>

    </div>
  )
}