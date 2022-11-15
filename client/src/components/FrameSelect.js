

export default function FrameSelect(props) {

    const srcs = props.srcs;
    const selectFunction = props.selectFunction;

    // return a grid of four images with hover effects
    return (
        <div className="frameSelect">
            {srcs.map((src, index) => (
                <div className="frameContainer" key={index} onClick={() => selectFunction(index)}>
                    <img
                        src={src}
                        alt={`Frame ${index}`}
                        className="frameImage"
                    />
                    <div className="frameOverlay">
                        <div className="text">Select Frame</div>
                    </div>
                </div>
            ))}
        </div>
    );

}