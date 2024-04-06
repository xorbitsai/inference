const boxDivStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr',
    columnGap: '20px',
    margin: '30px 2rem'
};

const PanelStyle = {
    tabPanelStyle: {
        padding: '0'
    },
    boxStyle: {
        margin: '20px'
    },
    boxDivStyle: boxDivStyle,
    box2ColDivStyle: {
        ...boxDivStyle,
        gridTemplateColumns: '150px 1fr',
    },
    cardsGridStyle: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
        paddingLeft: '2rem',
        gridGap: '2rem 0rem'
    }
};

export default PanelStyle