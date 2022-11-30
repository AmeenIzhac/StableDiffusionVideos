

export default function DropdownOptions({
  isImg2Img,
  setisImg2Img,
}) {
  const dark = '#181818'
  const light = '#9A4EAE'
  const defaultStyle = {
      backgroundColor: light,
      color: dark
  }
  const chosenStyle = {
      backgroundColor: 'rgba(24,24,24,0.8)',
      color:light,
      borderTop: '1px solid ' + light
  }

  const img2imgStyle = isImg2Img ? chosenStyle : defaultStyle 
  const walkStyle = !isImg2Img ? chosenStyle : defaultStyle 

  const chooseImg2Img = () => {
        setisImg2Img(true)
    }
    const chooseWalk = () => {
        setisImg2Img(false)
    }
    return (
      <div className="dropdownVidOption fullRow">
          <ul className="vidOptionContainer">
              <li className="img2img">
                  <button className="vidOptionBtn" style={img2imgStyle} onClick={chooseImg2Img}>img2img</button>
              </li>
              <li className="walk">
                  <button className="vidOptionBtn" style={walkStyle} onClick={chooseWalk}>Walk</button>
              </li>
          </ul>
      </div>
    )

}