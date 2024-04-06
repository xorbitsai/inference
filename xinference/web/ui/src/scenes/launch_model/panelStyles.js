const boxDivStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr',
    columnGap: '20px'
};

const PanelStyle = {
    tabPanelStyle: {
        padding: '0'
    },
    boxStyle: {
        m: '20px',
        padding: '30px 2rem'
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